"""
src/security.py — Security Utilities
=====================================
Handles all security-related operations:
  - File type validation (magic bytes, not just extension)
  - File size enforcement
  - Filename sanitization
  - Rate limiting (in-memory, per session)
  - Malicious content checks
"""

import os
import re
import time
import hashlib
import imghdr
import logging
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_EXTENSIONS       = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Magic bytes (file signatures) for supported types
MAGIC_BYTES = {
    # JPEG
    b"\xff\xd8\xff": "image/jpeg",
    # PNG
    b"\x89PNG": "image/png",
    # WebP
    b"RIFF": "image/webp",
    # MP4
    b"\x00\x00\x00\x18ftyp": "video/mp4",
    b"\x00\x00\x00\x1cftyp": "video/mp4",
    # AVI
    b"RIFF....AVI ": "video/avi",
    # MOV (QuickTime)
    b"\x00\x00\x00\x14ftyp": "video/quicktime",
}

MAX_FILE_SIZE_BYTES  = 50 * 1024 * 1024   # 50 MB
MAX_REQUESTS_PER_MIN = 10                  # Per session
MAX_FILENAME_LENGTH  = 100


# -----------------------------------------------------------------------
# Rate Limiter (simple in-memory, session-based)
# -----------------------------------------------------------------------
class RateLimiter:
    """
    Simple token-bucket rate limiter.
    Tracks request timestamps per session ID.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = defaultdict(list)

    def is_allowed(self, session_id: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed for the given session.

        Returns:
            (allowed: bool, remaining: int)
        """
        now      = time.time()
        window   = now - self.window_seconds
        history  = self._requests[session_id]

        # Remove timestamps outside the current window
        self._requests[session_id] = [t for t in history if t > window]

        remaining = self.max_requests - len(self._requests[session_id])

        if len(self._requests[session_id]) >= self.max_requests:
            return False, 0

        self._requests[session_id].append(now)
        return True, remaining - 1

    def reset(self, session_id: str):
        """Clear rate limit history for a session."""
        self._requests.pop(session_id, None)


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=MAX_REQUESTS_PER_MIN, window_seconds=60)


# -----------------------------------------------------------------------
# File Validation
# -----------------------------------------------------------------------
def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent path traversal and injection.

    - Removes directory components (../../etc)
    - Removes special characters
    - Truncates to max length
    - Preserves extension

    Returns:
        Safe filename string.
    """
    # Strip path components
    filename = Path(filename).name

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Keep only safe characters: alphanumeric, dots, dashes, underscores
    name, *ext_parts = filename.rsplit(".", 1)
    safe_name = re.sub(r"[^\w\-]", "_", name)

    if ext_parts:
        safe_ext  = re.sub(r"[^\w]", "", ext_parts[0]).lower()
        safe_name = f"{safe_name}.{safe_ext}"

    # Truncate
    if len(safe_name) > MAX_FILENAME_LENGTH:
        ext   = Path(safe_name).suffix
        stem  = safe_name[: MAX_FILENAME_LENGTH - len(ext)]
        safe_name = stem + ext

    return safe_name or "upload"


def validate_file_size(file_bytes: bytes, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> Tuple[bool, str]:
    """
    Check that file does not exceed size limit.

    Returns:
        (valid: bool, message: str)
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if len(file_bytes) > max_size_bytes:
        return False, f"File too large: {size_mb:.1f} MB. Maximum allowed: {max_size_bytes / 1024 / 1024:.0f} MB."
    return True, f"File size OK: {size_mb:.2f} MB"


def validate_file_extension(filename: str) -> Tuple[bool, str]:
    """
    Validate file extension against allowlist.

    Returns:
        (valid: bool, message: str)
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            False,
            f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return True, f"Extension '{ext}' is allowed."


def validate_file_magic_bytes(file_bytes: bytes) -> Tuple[bool, str]:
    """
    Validate actual file type using magic bytes (not just extension).
    Prevents disguised malicious files.

    Returns:
        (valid: bool, message: str)
    """
    header = file_bytes[:16]

    for magic, mime_type in MAGIC_BYTES.items():
        if header.startswith(magic):
            return True, f"Valid file type: {mime_type}"

    # Fallback: use imghdr for images
    img_type = imghdr.what(None, h=file_bytes[:32])
    if img_type in ("jpeg", "png", "webp"):
        return True, f"Valid image type: {img_type}"

    return False, "File content does not match any allowed type. Upload may be malicious."


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file for deduplication/logging."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_upload(
    filename: str,
    file_bytes: bytes,
    session_id: str = "default",
) -> Tuple[bool, str, dict]:
    """
    Full validation pipeline for an uploaded file.

    Checks (in order):
        1. Rate limiting
        2. File size
        3. Extension allowlist
        4. Magic bytes (real content type)

    Args:
        filename   : Original filename from the upload.
        file_bytes : Raw bytes of the uploaded file.
        session_id : Session or user identifier for rate limiting.

    Returns:
        (is_valid: bool, message: str, metadata: dict)
    """
    metadata = {
        "original_filename": filename,
        "safe_filename"    : sanitize_filename(filename),
        "size_bytes"       : len(file_bytes),
        "sha256"           : compute_file_hash(file_bytes),
    }

    # 1. Rate limiting
    allowed, remaining = rate_limiter.is_allowed(session_id)
    if not allowed:
        return False, "⚠️ Rate limit exceeded. Please wait 1 minute before uploading again.", metadata

    # 2. File size
    ok, msg = validate_file_size(file_bytes)
    if not ok:
        return False, msg, metadata

    # 3. Extension
    ok, msg = validate_file_extension(filename)
    if not ok:
        return False, msg, metadata

    # 4. Magic bytes
    ok, msg = validate_file_magic_bytes(file_bytes)
    if not ok:
        logger.warning(f"[SECURITY] Magic byte validation failed for '{filename}' | SHA256: {metadata['sha256']}")
        return False, f"❌ {msg}", metadata

    metadata["file_type"] = "video" if Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS else "image"
    logger.info(f"[SECURITY] File validated OK: {metadata['safe_filename']} ({metadata['size_bytes']} bytes)")
    return True, "✅ File validated successfully.", metadata


# -----------------------------------------------------------------------
# Environment Security
# -----------------------------------------------------------------------
def check_environment_security() -> list:
    """
    Check for common security misconfigurations at startup.

    Returns:
        List of warning strings (empty if all good).
    """
    warnings = []

    # Check that .env is not publicly accessible
    if os.path.exists(".env"):
        env_stat = os.stat(".env")
        # On Unix: check permissions (skip on Windows)
        if hasattr(env_stat, "st_mode"):
            import stat
            mode = env_stat.st_mode
            if mode & stat.S_IROTH:
                warnings.append("⚠️  .env file is world-readable. Run: chmod 600 .env")

    # Check for debug mode in production
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    app_env    = os.getenv("APP_ENV", "development")
    if debug_mode and app_env == "production":
        warnings.append("⚠️  DEBUG=True in production environment. Set DEBUG=False.")

    # Check that a secret key is set
    secret = os.getenv("APP_SECRET_KEY", "")
    if not secret or secret == "change_this_to_a_random_64_char_string":
        warnings.append("⚠️  APP_SECRET_KEY is not set or using default. Set a strong random key in .env.")

    return warnings
