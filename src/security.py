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
import io
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import defaultdict
from PIL import Image

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_EXTENSIONS       = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Magic bytes (file signatures) for supported types
MAGIC_BYTES = {
    # JPEG
    b"\xff\xd8\xff": "image/jpeg",
    # PNG
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\x89PNG": "image/png",
    # WebP / AVI (RIFF container)
    b"RIFF": "image/webp/video/avi",
    # MP4 / MOV / MKV (ISO Base Media file format)
    b"\x00\x00\x00": "video/mp4",
    b"\x1a\x45\xdf\xa3": "video/webm/mkv",
}

MAX_FILE_SIZE_BYTES  = 50 * 1024 * 1024   # 50 MB
MAX_REQUESTS_PER_MIN = 20                  # Per session
MAX_FILENAME_LENGTH  = 100


# -----------------------------------------------------------------------
# Rate Limiter (token bucket, session-based)
# -----------------------------------------------------------------------
class RateLimiter:
    """
    In-memory session rate limiter.
    Tracks request timestamps per session ID.
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests   = max_requests
        self.window_seconds = window_seconds
        self._requests: dict = defaultdict(list)

    def is_allowed(self, session_id: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed for the given session.

        Returns:
            (allowed: bool, remaining_requests: int)
        """
        now     = time.time()
        window  = now - self.window_seconds
        history = self._requests[session_id]

        # Evict timestamps outside the sliding window
        self._requests[session_id] = [t for t in history if t > window]

        remaining = self.max_requests - len(self._requests[session_id])

        if len(self._requests[session_id]) >= self.max_requests:
            return False, 0

        self._requests[session_id].append(now)
        return True, max(0, remaining - 1)

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
    - Preserves valid extension

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
        ext  = Path(safe_name).suffix
        stem = safe_name[: MAX_FILENAME_LENGTH - len(ext)]
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
            f"File type '{ext}' not allowed. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    return True, f"Extension '{ext}' is allowed."


def validate_file_magic_bytes(file_bytes: bytes, ext: str) -> Tuple[bool, str]:
    """
    Validate actual file type using magic bytes and PIL verification.
    Prevents disguised malicious files.

    Returns:
        (valid: bool, message: str)
    """
    header = file_bytes[:16]

    for magic, mime_type in MAGIC_BYTES.items():
        if header.startswith(magic):
            return True, f"Valid file signature: {mime_type}"

    # If it's an image, attempt PIL parse as a verification
    if ext in ALLOWED_IMAGE_EXTENSIONS:
        try:
            with Image.open(io.BytesIO(file_bytes)) as img:
                img.verify()
            return True, "Valid image content verified."
        except Exception:
            return False, "Corrupted or invalid image structure."

    if ext in ALLOWED_VIDEO_EXTENSIONS:
        # Video containers start with standard sync words or ftpy tags
        if b"ftyp" in header or header.startswith(b"\x00\x00\x00") or header.startswith(b"RIFF"):
            return True, "Valid video container."

    return False, "File content does not match allowed binary signature."


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file for integrity/logging."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_upload(
    filename: str,
    file_bytes: bytes,
    session_id: str = "default",
) -> Tuple[bool, str, dict]:
    """
    Full validation pipeline for an uploaded file.

    Checks:
        1. Rate limiting
        2. File size
        3. Extension allowlist
        4. Magic bytes & content structure

    Args:
        filename   : Original filename from the upload.
        file_bytes : Raw bytes of the uploaded file.
        session_id : Session or user identifier for rate limiting.

    Returns:
        (is_valid: bool, message: str, metadata: dict)
    """
    ext = Path(filename).suffix.lower()
    metadata = {
        "original_filename": filename,
        "safe_filename"    : sanitize_filename(filename),
        "size_bytes"       : len(file_bytes),
        "size_mb"          : round(len(file_bytes) / (1024 * 1024), 2),
        "sha256"           : compute_file_hash(file_bytes),
        "ext"              : ext,
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
    ok, msg = validate_file_magic_bytes(file_bytes, ext)
    if not ok:
        logger.warning(f"[SECURITY] Magic byte validation failed for '{filename}' | SHA256: {metadata['sha256']}")
        return False, f"❌ {msg}", metadata

    metadata["file_type"] = "video" if ext in ALLOWED_VIDEO_EXTENSIONS else "image"
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

    # Check that .env is not world-readable
    if os.path.exists(".env"):
        env_stat = os.stat(".env")
        if hasattr(env_stat, "st_mode") and os.name != "nt":
            import stat
            if env_stat.st_mode & stat.S_IROTH:
                warnings.append("⚠️  .env file is world-readable. Run: chmod 600 .env")

    # Check for debug mode in production
    debug_mode = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")
    app_env    = os.getenv("APP_ENV", "development")
    if debug_mode and app_env == "production":
        warnings.append("⚠️  DEBUG=True in production environment. Set DEBUG=False.")

    return warnings
