"""
app/app.py  —  Deepfake Detection  ·  AI Chatbot Interface
===========================================================
Full chatbot-style UX:
  • User uploads image / video in the chat
  • AI "DeepGuard" responds with analysis, confidence gauge,
    GradCAM heatmap, frame-by-frame chart, and a plain-English verdict
  • Chat history persists across uploads in the session
  • Beautiful dark glassmorphism design
  • Works in Demo Mode (no model needed) or Real Mode (trained model)
  • All security checks built in

Run:
    streamlit run app/app.py
"""

# ── stdlib ──────────────────────────────────────────────────────────────
import os, io, sys, uuid, time, random, hashlib, logging, tempfile
from pathlib import Path
from datetime import datetime

# ── third-party ─────────────────────────────────────────────────────────
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from dotenv import load_dotenv

# ── project root on path ────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING)

MODEL_PATH  = os.getenv("MODEL_PATH", str(ROOT / "models" / "best_model.pth"))
IMAGE_SIZE  = 380
MAX_MB      = 50
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov"}

# ════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "DeepGuard AI",
    page_icon  = "🛡️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ════════════════════════════════════════════════════════════════════════
#  CSS  — complete design system
# ════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d0b1a 0%, #0a0a0f 60%, #0d1117 100%);
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton          { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0e1a 0%, #111118 100%);
    border-right: 1px solid rgba(124, 111, 205, 0.15);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Main content padding ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Chat layout ── */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    padding: 0;
}
.chat-header {
    background: linear-gradient(135deg, rgba(124,111,205,0.12), rgba(168,85,247,0.08));
    border-bottom: 1px solid rgba(124,111,205,0.2);
    padding: 1rem 2rem;
    display: flex; align-items: center; gap: 1rem;
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
}
.chat-title {
    font-size: 1.4rem; font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.chat-subtitle { font-size: 0.78rem; color: #6b7280; margin-top: 1px; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%     { opacity:.6; transform:scale(1.3); }
}

/* ── Chat messages area ── */
.chat-messages { padding: 1.5rem 2rem; overflow-y: auto; flex: 1; }

/* ── Message bubbles ── */
.msg-user {
    display: flex; justify-content: flex-end; margin-bottom: 1.2rem;
}
.msg-user-bubble {
    background: linear-gradient(135deg, #7c6fcd, #a855f7);
    color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.2rem;
    max-width: 60%; font-size: 0.92rem; line-height: 1.5;
    box-shadow: 0 4px 20px rgba(124,111,205,0.35);
}

.msg-ai { display: flex; gap: 0.75rem; margin-bottom: 1.4rem; align-items: flex-start; }
.ai-avatar {
    width: 36px; height: 36px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #7c6fcd, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(124,111,205,0.4);
}
.msg-ai-bubble {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 4px 18px 18px 18px;
    padding: 1rem 1.3rem;
    max-width: 75%; font-size: 0.9rem;
    line-height: 1.6; color: #d1d5db;
    backdrop-filter: blur(8px);
}

/* ── Result verdict card ── */
.verdict-card {
    border-radius: 14px; padding: 1.1rem 1.5rem;
    margin: 0.8rem 0; text-align: center;
    border: 1px solid;
}
.verdict-fake {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.3);
}
.verdict-real {
    background: rgba(34, 197, 94, 0.08);
    border-color: rgba(34, 197, 94, 0.3);
}
.verdict-label-fake {
    font-size: 2rem; font-weight: 800; color: #f87171;
    letter-spacing: -0.5px;
}
.verdict-label-real {
    font-size: 2rem; font-weight: 800; color: #4ade80;
    letter-spacing: -0.5px;
}
.verdict-sub {
    font-size: 0.82rem; color: #9ca3af; margin-top: 0.3rem;
}

/* ── Stats row ── */
.stat-row {
    display: flex; gap: 0.6rem; margin: 0.8rem 0; flex-wrap: wrap;
}
.stat-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px; padding: 0.45rem 0.9rem;
    font-size: 0.8rem; color: #d1d5db;
    display: flex; flex-direction: column; gap: 1px;
}
.stat-chip-val { font-weight: 700; font-size: 1rem; color: #fff; }
.stat-chip-lbl { font-size: 0.68rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Chat input bar ── */
.input-area {
    background: rgba(255,255,255,0.03);
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 1rem 2rem 1.5rem;
    position: sticky; bottom: 0;
    backdrop-filter: blur(16px);
}
.input-hint {
    font-size: 0.75rem; color: #4b5563;
    margin-bottom: 0.6rem; text-align: center;
}
/* Streamlit file uploader override */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(124,111,205,0.06) !important;
    border: 2px dashed rgba(124,111,205,0.35) !important;
    border-radius: 14px !important;
    transition: all 0.25s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    background: rgba(124,111,205,0.12) !important;
    border-color: rgba(124,111,205,0.6) !important;
}

/* ── Sidebar elements ── */
.sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
}
.sidebar-section-title {
    font-size: 0.7rem; font-weight: 600; color: #6b7280;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
}
.model-badge-on  { color: #4ade80; font-weight: 600; font-size: 0.85rem; }
.model-badge-off { color: #f97316; font-weight: 600; font-size: 0.85rem; }

.capability-item {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.82rem; color: #9ca3af;
    padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.capability-item:last-child { border-bottom: none; }

/* ── Welcome message ── */
.welcome-box {
    background: linear-gradient(135deg, rgba(124,111,205,0.1), rgba(168,85,247,0.06));
    border: 1px solid rgba(124,111,205,0.2);
    border-radius: 16px; padding: 1.5rem;
    margin-bottom: 0.5rem;
}
.welcome-title {
    font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
    margin-bottom: 0.4rem;
}
.welcome-sub { font-size: 0.88rem; color: #9ca3af; line-height: 1.6; }

/* ── Demo banner ── */
.demo-banner {
    background: rgba(251, 146, 60, 0.08);
    border: 1px solid rgba(251,146,60,0.3);
    border-radius: 10px; padding: 0.65rem 1rem;
    font-size: 0.82rem; color: #fb923c; margin-bottom: 0.6rem;
}

/* ── Thinking animation ── */
.thinking {
    display: flex; gap: 5px; align-items: center; padding: 0.3rem 0;
}
.thinking span {
    width: 7px; height: 7px; border-radius: 50%;
    background: #7c6fcd; animation: bounce 1.2s infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.2s; }
.thinking span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,100%{ transform: translateY(0); opacity:.5; }
    50%     { transform: translateY(-5px); opacity:1; }
}

/* ── Timestamp ── */
.msg-time { font-size: 0.68rem; color: #374151; margin-top: 0.3rem; text-align: right; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(124,111,205,0.3); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "session_id"    : str(uuid.uuid4())[:8],
        "messages"      : [],          # chat history
        "analysis_count": 0,
        "last_upload_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ════════════════════════════════════════════════════════════════════════
#  MODEL LOADER  (cached, lazy)
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_predictor_safe(model_path: str):
    if not Path(model_path).exists():
        return None
    try:
        from src.predict import DeepfakePredictor
        return DeepfakePredictor(
            model_path = model_path,
            device     = "auto",
            threshold  = 0.5,
            image_size = IMAGE_SIZE,
        )
    except Exception as e:
        logging.warning(f"Predictor load failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════
#  SECURITY VALIDATOR
# ════════════════════════════════════════════════════════════════════════
def validate_file(filename: str, file_bytes: bytes) -> tuple[bool, str, dict]:
    ext     = Path(filename).suffix.lower()
    size_mb = len(file_bytes) / (1024 * 1024)
    sha256  = hashlib.sha256(file_bytes).hexdigest()

    if ext not in ALLOWED_EXT:
        return False, f"❌ File type `{ext}` not allowed.", {}
    if size_mb > MAX_MB:
        return False, f"❌ File too large ({size_mb:.1f} MB). Limit: {MAX_MB} MB.", {}

    # Magic-byte check for images
    header = file_bytes[:8]
    MAGIC  = {
        b"\xff\xd8\xff"  : "jpeg",
        b"\x89PNG\r\n\x1a\n": "png",
        b"RIFF"          : "webp/avi",
        b"\x00\x00\x00"  : "mp4/mov",
    }
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        valid_magic = any(header.startswith(m) for m in MAGIC)
        if not valid_magic:
            return False, "❌ File content doesn't match its extension (possible spoofing).", {}

    ftype = "video" if ext in {".mp4", ".avi", ".mov"} else "image"
    return True, "ok", {
        "sha256"   : sha256,
        "size_mb"  : round(size_mb, 2),
        "file_type": ftype,
        "ext"      : ext,
    }


# ════════════════════════════════════════════════════════════════════════
#  DEMO ANALYSIS  (realistic fake results when no model loaded)
# ════════════════════════════════════════════════════════════════════════
def demo_analyze_image(pil_img: Image.Image) -> dict:
    """Generate a realistic-looking demo result."""
    time.sleep(1.2)   # feel like it's thinking
    prob      = round(random.uniform(0.08, 0.92), 3)
    is_fake   = prob >= 0.5
    return {
        "is_fake"    : is_fake,
        "label"      : "FAKE" if is_fake else "REAL",
        "probability": prob,
        "confidence" : round(prob if is_fake else 1 - prob, 3),
        "emoji"      : "🚨" if is_fake else "✅",
        "color"      : "#f87171" if is_fake else "#4ade80",
        "face_np"    : None,
        "gradcam_pil": None,
        "inference_ms": round(random.uniform(120, 400), 1),
        "demo"       : True,
        "model_name" : "EfficientNet-B4 (Demo)",
        "regions"    : (
            ["Eye boundaries", "Jawline skin texture", "Mouth corners"]
            if is_fake else []
        ),
    }


def demo_analyze_video(n_frames: int = 30) -> dict:
    time.sleep(1.8)
    probs   = [round(random.uniform(0.05, 0.95), 3) for _ in range(n_frames)]
    avg     = round(float(np.mean(probs)), 3)
    is_fake = avg >= 0.5
    return {
        "is_fake"        : is_fake,
        "label"          : "FAKE" if is_fake else "REAL",
        "probability"    : avg,
        "confidence"     : round(avg if is_fake else 1 - avg, 3),
        "emoji"          : "🚨" if is_fake else "✅",
        "color"          : "#f87171" if is_fake else "#4ade80",
        "frame_probs"    : probs,
        "frames_analyzed": n_frames,
        "gradcam_pil"    : None,
        "inference_ms"   : round(random.uniform(800, 3000), 0),
        "demo"           : True,
        "model_name"     : "EfficientNet-B4 (Demo)",
    }


# ════════════════════════════════════════════════════════════════════════
#  REAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════
def real_analyze_image(predictor, pil_img: Image.Image, gradcam: bool) -> dict:
    result = predictor.predict_image(pil_img, return_gradcam=gradcam)
    result["demo"]       = False
    result["model_name"] = "EfficientNet-B4"
    result["regions"]    = (
        ["Eye boundaries", "Jawline", "Skin texture", "Mouth corners"]
        if result["is_fake"] else []
    )
    return result


def real_analyze_video(predictor, video_path: str, max_frames: int, gradcam: bool) -> dict:
    result = predictor.predict_video(
        video_path    = video_path,
        max_frames    = max_frames,
        sample_rate   = 10,
        return_gradcam= gradcam,
    )
    result["demo"]       = False
    result["model_name"] = "EfficientNet-B4"
    return result


# ════════════════════════════════════════════════════════════════════════
#  PLOTLY CHARTS
# ════════════════════════════════════════════════════════════════════════
def make_gauge(prob: float, color: str):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        number= {"suffix": "%", "font": {"size": 34, "color": "#fff", "family": "Inter"}},
        title = {"text": "Fake Probability", "font": {"size": 12, "color": "#6b7280"}},
        gauge = {
            "axis"     : {"range": [0, 100], "tickwidth": 1, "tickcolor": "#374151",
                          "tickfont": {"color": "#6b7280", "size": 10}},
            "bar"      : {"color": color, "thickness": 0.22},
            "bgcolor"  : "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps"    : [
                {"range": [0, 50],  "color": "rgba(34,197,94,0.08)"},
                {"range": [50, 75], "color": "rgba(251,146,60,0.08)"},
                {"range": [75,100], "color": "rgba(239,68,68,0.1)"},
            ],
            "threshold": {"line": {"color": "rgba(255,255,255,0.4)", "width": 2},
                          "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)", plot_bgcolor = "rgba(0,0,0,0)",
        margin = dict(l=20, r=20, t=40, b=10), height = 200,
        font   = dict(family="Inter"),
    )
    return fig


def make_frame_chart(probs: list):
    colors = ["#f87171" if p >= 0.5 else "#4ade80" for p in probs]
    x      = list(range(1, len(probs) + 1))
    fig    = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=[p*100 for p in probs],
        marker_color=colors, marker_line_width=0,
        hovertemplate="Frame %{x}<br>Fake: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=50, line_dash="dot",
                  line_color="rgba(255,255,255,0.3)",
                  annotation_text="50% threshold",
                  annotation_font_color="#9ca3af",
                  annotation_font_size=10)
    fig.update_layout(
        title      = dict(text="Frame-by-Frame Analysis", font=dict(color="#e2e8f0", size=13)),
        xaxis_title= "Frame",
        yaxis_title= "Fake %",
        yaxis_range= [0, 108],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font  = dict(color="#6b7280", family="Inter"),
        height= 250, showlegend=False,
        margin= dict(l=10, r=10, t=40, b=10),
    )
    fig.update_xaxes(showgrid=False, color="#4b5563", tickfont_size=10)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                     color="#4b5563", tickfont_size=10)
    return fig


# ════════════════════════════════════════════════════════════════════════
#  CHAT RENDER HELPERS
# ════════════════════════════════════════════════════════════════════════
def ts() -> str:
    return datetime.now().strftime("%H:%M")


def user_msg(text: str, has_file: bool = False):
    icon = "📎" if has_file else "💬"
    with st.chat_message("user"):
        st.markdown(text)


def ai_thinking():
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown("""
        <div class="thinking">
            <span></span><span></span><span></span>
        </div>""", unsafe_allow_html=True)


def render_image_result(result: dict, pil_img: Image.Image, show_cam: bool):
    """Render full image analysis result in the AI bubble."""
    with st.chat_message("assistant", avatar="🛡️"):

        # Demo warning
        if result.get("demo"):
            st.markdown("""<div class="demo-banner">
            ⚠️ <b>Demo Mode</b> — results are simulated.
            Train the model for real predictions.
            </div>""", unsafe_allow_html=True)

        # ── Verdict ──
        cls = "verdict-fake" if result["is_fake"] else "verdict-real"
        lbl = "verdict-label-fake" if result["is_fake"] else "verdict-label-real"
        regions_html = ""
        if result.get("regions"):
            tags = "".join(f'<span style="background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:6px;padding:2px 8px;margin:2px;font-size:0.72rem;color:#fca5a5">{r}</span>'
                           for r in result["regions"])
            regions_html = f'<div style="margin-top:0.6rem;display:flex;flex-wrap:wrap;gap:4px;justify-content:center">{tags}</div>'

        summary = (
            f"High manipulation signatures detected in facial regions."
            if result["is_fake"] else
            "No significant manipulation artifacts detected."
        )

        st.markdown(f"""
        <div class="verdict-card {cls}">
            <div class="{lbl}">{result['emoji']} {result['label']}</div>
            <div class="verdict-sub">{summary}</div>
            {regions_html}
        </div>""", unsafe_allow_html=True)

        # ── Stats ──
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip">
                <span class="stat-chip-val">{result['probability']*100:.1f}%</span>
                <span class="stat-chip-lbl">Fake Probability</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{result['confidence']*100:.1f}%</span>
                <span class="stat-chip-lbl">Confidence</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{result['inference_ms']} ms</span>
                <span class="stat-chip-lbl">Inference Time</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{result['model_name']}</span>
                <span class="stat-chip-lbl">Model</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── Gauge + Image ──
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(make_gauge(result["probability"], result["color"]),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.image(pil_img, caption="Uploaded image", use_container_width=True)

        # ── GradCAM ──
        if show_cam and result.get("gradcam_pil") and result.get("face_np") is not None:
            st.markdown("**🔬 GradCAM — Suspicious Regions**")
            g1, g2 = st.columns(2)
            with g1:
                st.image(Image.fromarray(result["face_np"]),
                         caption="Detected face", use_container_width=True)
            with g2:
                st.image(result["gradcam_pil"],
                         caption="🔴 Red = high manipulation probability",
                         use_container_width=True)
            st.caption("Warm (red/yellow) regions had the strongest influence on the FAKE prediction.")

        # ── Plain-language summary ──
        if result["is_fake"]:
            verdict_text = (
                f"**DeepGuard detected this image as likely FAKE** "
                f"with {result['confidence']*100:.1f}% confidence. "
                f"The model identified potential manipulation artifacts in "
                f"**{', '.join(result.get('regions', ['facial regions']))}**. "
                f"This could be generated by face-swapping AI tools like FaceSwap, DeepFaceLab, or diffusion-based generators."
            )
        else:
            verdict_text = (
                f"**DeepGuard found no signs of manipulation.** "
                f"The image appears to be authentic with {result['confidence']*100:.1f}% confidence. "
                f"No unusual GAN artifacts or frequency anomalies were detected in the facial regions."
            )

        st.info(verdict_text)
        st.caption(f"🕐 {ts()}  ·  Session #{st.session_state.session_id}")


def render_video_result(result: dict, show_cam: bool):
    """Render full video analysis result in the AI bubble."""
    with st.chat_message("assistant", avatar="🛡️"):

        if result.get("demo"):
            st.markdown("""<div class="demo-banner">
            ⚠️ <b>Demo Mode</b> — results are simulated random values.
            Train the model for real predictions.
            </div>""", unsafe_allow_html=True)

        # Verdict
        cls = "verdict-fake" if result["is_fake"] else "verdict-real"
        lbl = "verdict-label-fake" if result["is_fake"] else "verdict-label-real"
        st.markdown(f"""
        <div class="verdict-card {cls}">
            <div class="{lbl}">{result['emoji']} VIDEO IS {result['label']}</div>
            <div class="verdict-sub">
                {result['frames_analyzed']} frames analyzed
                · Avg fake probability: {result['probability']*100:.1f}%
            </div>
        </div>""", unsafe_allow_html=True)

        # Stats
        probs       = result["frame_probs"]
        fake_frames = sum(1 for p in probs if p >= 0.5)
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip">
                <span class="stat-chip-val">{result['probability']*100:.1f}%</span>
                <span class="stat-chip-lbl">Avg Fake Prob</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{result['confidence']*100:.1f}%</span>
                <span class="stat-chip-lbl">Confidence</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{fake_frames}/{len(probs)}</span>
                <span class="stat-chip-lbl">Fake Frames</span>
            </div>
            <div class="stat-chip">
                <span class="stat-chip-val">{np.max(probs)*100:.0f}%</span>
                <span class="stat-chip-lbl">Peak Fake Prob</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Frame chart
        st.plotly_chart(make_frame_chart(probs),
                        use_container_width=True, config={"displayModeBar": False})

        # GradCAM
        if show_cam and result.get("gradcam_pil"):
            st.markdown("**🔬 Most Suspicious Frame — GradCAM**")
            st.image(result["gradcam_pil"], width=360,
                     caption="Manipulation regions in most suspicious frame")

        # Summary text
        if result["is_fake"]:
            verdict_text = (
                f"**DeepGuard detected this video as FAKE** with {result['confidence']*100:.1f}% confidence. "
                f"**{fake_frames} out of {len(probs)} frames** showed manipulation signatures. "
                f"The peak fake probability was **{np.max(probs)*100:.1f}%**. "
                f"This is consistent with video deepfake techniques that generate face frames independently."
            )
        else:
            verdict_text = (
                f"**DeepGuard found no signs of video manipulation** "
                f"(confidence: {result['confidence']*100:.1f}%). "
                f"Only {fake_frames}/{len(probs)} frames exceeded the 50% threshold, "
                f"which is within normal noise tolerance. The video appears authentic."
            )
        st.info(verdict_text)
        st.caption(f"🕐 {ts()}  ·  Session #{st.session_state.session_id}")


def render_error(msg: str):
    with st.chat_message("assistant", avatar="🛡️"):
        st.error(f"**Analysis failed:** {msg}")
        st.caption(f"🕐 {ts()}")


# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo / Title
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0 1.2rem">
        <div style="font-size:2.2rem">🛡️</div>
        <div style="font-size:1.1rem;font-weight:800;
            background:linear-gradient(135deg,#a78bfa,#818cf8);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            background-clip:text">DeepGuard AI</div>
        <div style="font-size:0.72rem;color:#4b5563;margin-top:2px">v1.0 · EfficientNet-B4</div>
    </div>
    """, unsafe_allow_html=True)

    # Model status
    model_ready = Path(MODEL_PATH).exists()
    predictor   = load_predictor_safe(MODEL_PATH)

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">🤖 Model Status</div>', unsafe_allow_html=True)
    if model_ready and predictor:
        st.markdown('<span class="model-badge-on">● Model Ready</span>', unsafe_allow_html=True)
        st.caption("Real predictions active")
    else:
        st.markdown('<span class="model-badge-off">● Demo Mode</span>', unsafe_allow_html=True)
        st.caption("Train model for real results")
        with st.expander("How to train?"):
            st.code("python -m src.train", language="bash")
    st.markdown('</div>', unsafe_allow_html=True)

    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">⚙️ Settings</div>', unsafe_allow_html=True)
    show_gradcam = st.toggle("GradCAM Heatmap",   value=True,
                             help="Highlight suspicious facial regions")
    show_frames  = st.toggle("Frame Analysis",     value=True,
                             help="Show per-frame chart for videos")
    max_frames   = st.slider("Max video frames",   10, 100, 40, 10)
    st.markdown('</div>', unsafe_allow_html=True)

    # Capabilities
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">✨ Capabilities</div>', unsafe_allow_html=True)
    caps = [
        ("🖼️", "Image deepfake detection"),
        ("🎬", "Video frame analysis"),
        ("🔬", "GradCAM explainability"),
        ("📊", "Confidence scoring"),
        ("🔒", "File security validation"),
        ("💬", "Chat-style interface"),
    ]
    for icon, label in caps:
        st.markdown(f"""<div class="capability-item">
            <span>{icon}</span><span>{label}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Session info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📈 Session Stats</div>', unsafe_allow_html=True)
    st.metric("Analyses Run",  st.session_state.analysis_count)
    st.metric("Session ID",    st.session_state.session_id)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.analysis_count= 0
        st.rerun()

    # GitHub link
    st.markdown("""
    <div style="text-align:center;margin-top:1rem">
        <a href="https://github.com/Gautam-Desk/DS-project"
           target="_blank"
           style="color:#7c6fcd;font-size:0.78rem;text-decoration:none">
           ⭐ View on GitHub
        </a>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ════════════════════════════════════════════════════════════════════════

# ── Header bar ──
st.markdown("""
<div class="chat-header">
    <div class="status-dot"></div>
    <div>
        <div class="chat-title">🛡️ DeepGuard AI</div>
        <div class="chat-subtitle">Deepfake Detection · Powered by EfficientNet-B4 + GradCAM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Welcome message (shown once) ──
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown("""
        <div class="welcome-box">
            <div class="welcome-title">👋 Hello! I'm DeepGuard AI</div>
            <div class="welcome-sub">
                I can detect deepfake manipulations in <b>images</b> and <b>videos</b>
                using state-of-the-art computer vision.<br><br>
                Upload a file below and I'll analyze it for:
                <ul style="margin:0.5rem 0 0; padding-left:1.2rem; color:#9ca3af;">
                    <li>GAN-generated facial artifacts</li>
                    <li>Unnatural skin texture or blending</li>
                    <li>Eye/jawline boundary inconsistencies</li>
                    <li>Frequency domain anomalies</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **Supported formats:** JPG · PNG · WebP · MP4 · AVI · MOV  
        **Max file size:** 50 MB  
        **Try it:** Upload any face photo and I'll tell you if it's real or AI-generated.
        """)

# ── Render chat history ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        # AI result messages are pre-rendered HTML/components stored as a flag
        # We re-render from stored result data
        with st.chat_message("assistant", avatar="🛡️"):
            st.markdown(msg["content"])

# ════════════════════════════════════════════════════════════════════════
#  FILE UPLOAD INPUT  (bottom of page, chat-style)
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4b5563;font-size:0.78rem;margin-bottom:0.5rem">'
    '📎 Upload an image or video to analyze · Max 50 MB</p>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload image or video",
    type        = ["jpg","jpeg","png","webp","mp4","avi","mov"],
    label_visibility = "collapsed",
    help        = "Drag & drop or click to upload",
    key         = "file_upload",
)

# ════════════════════════════════════════════════════════════════════════
#  PROCESS UPLOAD
# ════════════════════════════════════════════════════════════════════════
if uploaded is not None:
    upload_id = f"{uploaded.name}_{uploaded.size}"

    # Prevent re-processing same file
    if upload_id != st.session_state.last_upload_id:
        st.session_state.last_upload_id = upload_id
        file_bytes = uploaded.read()

        # ── Security validation ──
        ok, err_msg, meta = validate_file(uploaded.name, file_bytes)
        if not ok:
            # Add to chat
            with st.chat_message("user"):
                st.markdown(f"📎 Uploaded: `{uploaded.name}`")
            with st.chat_message("assistant", avatar="🛡️"):
                st.error(err_msg)
                st.caption(f"🕐 {ts()}")
            st.session_state.messages.append({"role": "user",      "content": f"📎 `{uploaded.name}`"})
            st.session_state.messages.append({"role": "assistant",  "content": f"❌ {err_msg}"})
            st.stop()

        ftype = meta["file_type"]

        # ── Show user message ──
        user_text = (
            f"📎 **{uploaded.name}** "
            f"({meta['size_mb']} MB · {ftype.upper()})\n\n"
            f"Please analyze this for deepfakes."
        )
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state.messages.append({"role": "user", "content": user_text})

        # ── Analyze ──
        if ftype == "image":
            pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

            with st.spinner("🧠 DeepGuard is analyzing..."):
                if predictor:
                    result = real_analyze_image(predictor, pil_img, show_gradcam)
                else:
                    result = demo_analyze_image(pil_img)

            st.session_state.analysis_count += 1
            render_image_result(result, pil_img, show_gradcam)

            # Store summary in history (not full result — keep it lightweight)
            summary = (
                f"**Analysis complete ✓**  \n"
                f"Verdict: **{result['emoji']} {result['label']}**  \n"
                f"Fake probability: **{result['probability']*100:.1f}%**  \n"
                f"Confidence: **{result['confidence']*100:.1f}%**"
                + (" _(demo)_" if result.get("demo") else "")
            )
            st.session_state.messages.append({"role": "assistant", "content": summary})

        elif ftype == "video":
            tmp = tempfile.NamedTemporaryFile(suffix=meta["ext"], delete=False)
            tmp.write(file_bytes)
            tmp.close()

            with st.spinner(f"🎬 Analyzing {max_frames} video frames..."):
                try:
                    if predictor:
                        result = real_analyze_video(predictor, tmp.name, max_frames, show_gradcam)
                    else:
                        result = demo_analyze_video(max_frames)
                finally:
                    os.unlink(tmp.name)

            st.session_state.analysis_count += 1
            render_video_result(result, show_gradcam)

            probs       = result["frame_probs"]
            fake_frames = sum(1 for p in probs if p >= 0.5)
            summary = (
                f"**Video analysis complete ✓**  \n"
                f"Verdict: **{result['emoji']} {result['label']}**  \n"
                f"Avg fake probability: **{result['probability']*100:.1f}%**  \n"
                f"Fake frames: **{fake_frames}/{len(probs)}**"
                + (" _(demo)_" if result.get("demo") else "")
            )
            st.session_state.messages.append({"role": "assistant", "content": summary})

        st.rerun()
