"""
app/app.py — Streamlit Web Application
========================================
Deepfake Detection System — Interactive Web Demo

Features:
  - Image & Video upload with full security validation
  - Real-time deepfake prediction with confidence score
  - GradCAM heatmap visualization
  - Video frame-by-frame analysis chart
  - Rate limiting per session
  - Secure file handling (magic bytes, size limits)
  - Clean, professional UI

Run locally:
    streamlit run app/app.py

Deploy:
    Push to GitHub → connect to Streamlit Community Cloud
"""

import os
import sys
import io
import uuid
import logging
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security import validate_upload, check_environment_security
from src.predict import DeepfakePredictor

# -----------------------------------------------------------------------
# Startup
# -----------------------------------------------------------------------
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
IMAGE_SIZE = 380

# -----------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------
st.set_page_config(
    page_title  = "Deepfake Detector | AI Vision",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
    menu_items  = {
        "Get Help"   : "https://github.com/your-repo/deepfake-detection",
        "Report a bug": None,
        "About"      : "AI-powered deepfake detection using EfficientNet-B4 + GradCAM"
    }
)

# -----------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------
st.markdown("""
<style>
    /* ---- Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Background ---- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ---- Main container ---- */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* ---- Header ---- */
    .hero-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-top: 0.5rem;
    }

    /* ---- Result Cards ---- */
    .result-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-label-fake {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FC5C7D;
    }
    .result-label-real {
        font-size: 2.5rem;
        font-weight: 700;
        color: #43E97B;
    }
    .result-sub {
        font-size: 0.95rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }

    /* ---- Upload area ---- */
    .stFileUploader > div {
        background: rgba(255,255,255,0.05) !important;
        border: 2px dashed rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: rgba(255,255,255,0.5) !important;
    }

    /* ---- Info boxes ---- */
    .info-chip {
        display: inline-block;
        background: rgba(67, 233, 123, 0.15);
        color: #43E97B;
        border: 1px solid rgba(67,233,123,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .warn-chip {
        background: rgba(252, 92, 125, 0.15);
        color: #FC5C7D;
        border-color: rgba(252,92,125,0.3);
    }

    /* ---- Section header ---- */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.75rem;
        border-left: 3px solid #667eea;
        padding-left: 0.75rem;
    }

    /* ---- Metric boxes ---- */
    .metric-box {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ---- Divider ---- */
    hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ---- Streamlit button ---- */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: opacity 0.2s;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------
# Session State Init
# -----------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "predictor" not in st.session_state:
    st.session_state.predictor = None
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0


# -----------------------------------------------------------------------
# Model Loading (cached)
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor(model_path: str) -> "DeepfakePredictor | None":
    """Load predictor once and cache for all sessions."""
    if not Path(model_path).exists():
        return None
    try:
        return DeepfakePredictor(model_path=model_path, device="auto", threshold=0.5, image_size=IMAGE_SIZE)
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        return None


# -----------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------
def bytes_to_pil(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def make_gauge_chart(probability: float, label: str, color: str):
    """Plotly gauge chart for confidence display."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = probability * 100,
        number= {"suffix": "%", "font": {"size": 36, "color": "#fff"}},
        delta = {"reference": 50, "font": {"size": 14}},
        title = {"text": f"Fake Probability", "font": {"size": 14, "color": "#a0aec0"}},
        gauge = {
            "axis"     : {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
            "bar"      : {"color": color, "thickness": 0.25},
            "bgcolor"  : "rgba(255,255,255,0.05)",
            "borderwidth": 0,
            "steps"    : [
                {"range": [0,  50], "color": "rgba(67,233,123,0.15)"},
                {"range": [50, 75], "color": "rgba(255,165,0,0.15)"},
                {"range": [75, 100],"color": "rgba(252,92,125,0.15)"},
            ],
            "threshold": {
                "line" : {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(l=30, r=30, t=40, b=10),
        height        = 240,
    )
    return fig


def make_frame_chart(frame_probs: list):
    """Plotly bar chart of per-frame fake probabilities."""
    frames = list(range(1, len(frame_probs) + 1))
    colors = ["#FC5C7D" if p >= 0.5 else "#43E97B" for p in frame_probs]

    fig = go.Figure(go.Bar(
        x           = frames,
        y           = [p * 100 for p in frame_probs],
        marker_color= colors,
        text        = [f"{p*100:.0f}%" for p in frame_probs],
        textposition= "outside",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="white", opacity=0.5,
                  annotation_text="Decision Threshold (50%)", annotation_position="top left")
    fig.update_layout(
        title        = "Per-Frame Fake Probability",
        xaxis_title  = "Frame Number",
        yaxis_title  = "Fake Probability (%)",
        yaxis_range  = [0, 110],
        paper_bgcolor= "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font         = dict(color="#a0aec0"),
        title_font   = dict(color="#fff", size=14),
        height       = 300,
        showlegend   = False,
    )
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", color="#555")
    return fig


# -----------------------------------------------------------------------
# HERO HEADER
# -----------------------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🔍 Deepfake Detector</div>
    <div class="hero-sub">AI-powered face manipulation detection using EfficientNet-B4 + GradCAM</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Security warnings (dev mode)
# -----------------------------------------------------------------------
sec_warnings = check_environment_security()
if sec_warnings:
    with st.expander("⚠️ Security Warnings", expanded=False):
        for w in sec_warnings:
            st.warning(w)

# -----------------------------------------------------------------------
# Load Model
# -----------------------------------------------------------------------
predictor = load_predictor(MODEL_PATH)

if predictor is None:
    st.error(
        "⚠️ **Model not found.**\n\n"
        f"Expected at: `{MODEL_PATH}`\n\n"
        "Please train the model first:\n"
        "```bash\npython -m src.train\n```"
    )
    st.info(
        "🎓 **Demo Mode**: The app UI is fully functional. "
        "Upload an image to see the interface — connect your trained model to get real predictions."
    )

st.divider()

# -----------------------------------------------------------------------
# UPLOAD SECTION
# -----------------------------------------------------------------------
col_upload, col_options = st.columns([3, 1])

with col_upload:
    st.markdown('<div class="section-title">📤 Upload Media</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label       = "Upload an image or video",
        type        = ["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
        help        = "Max size: 50 MB | Supported: JPG, PNG, WebP, MP4, AVI, MOV",
        label_visibility = "collapsed",
    )

with col_options:
    st.markdown('<div class="section-title">⚙️ Options</div>', unsafe_allow_html=True)
    show_gradcam = st.toggle("GradCAM Heatmap", value=True,
                             help="Highlight suspicious facial regions")
    show_raw_prob = st.toggle("Show Raw Scores", value=False,
                              help="Show all probability values")
    if uploaded and uploaded.name.lower().endswith(("mp4", "avi", "mov")):
        max_frames = st.slider("Max frames to analyze", 10, 100, 50, step=10)
    else:
        max_frames = 50

# -----------------------------------------------------------------------
# PROCESS UPLOAD
# -----------------------------------------------------------------------
if uploaded is not None:
    file_bytes  = uploaded.read()
    session_id  = st.session_state.session_id

    # --- Security Validation ---
    with st.spinner("🔒 Validating file security..."):
        is_valid, sec_msg, metadata = validate_upload(
            filename   = uploaded.name,
            file_bytes = file_bytes,
            session_id = session_id,
        )

    if not is_valid:
        st.error(f"**Security Check Failed:** {sec_msg}")
        st.stop()

    file_type = metadata.get("file_type", "image")
    st.markdown(f"""
    <div>
        <span class="info-chip">✅ File validated</span>
        <span class="info-chip">{metadata['size_bytes'] / 1024:.1f} KB</span>
        <span class="info-chip">{file_type.upper()}</span>
        <span class="info-chip">SHA256: {metadata['sha256'][:16]}...</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ----------------------------------------------------------------
    # IMAGE PREDICTION
    # ----------------------------------------------------------------
    if file_type == "image":
        pil_image = bytes_to_pil(file_bytes)

        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.markdown('<div class="section-title">🖼️ Input Image</div>', unsafe_allow_html=True)
            st.image(pil_image, use_column_width=True)

        with col_result:
            if predictor is None:
                st.info("🎓 Demo mode — train a model to see predictions.")
            else:
                with st.spinner("🧠 Analyzing..."):
                    result = predictor.predict_image(pil_image, return_gradcam=show_gradcam)
                    st.session_state.analysis_count += 1

                # --- Main result card ---
                label_class = "result-label-fake" if result["is_fake"] else "result-label-real"
                st.markdown(f"""
                <div class="result-card">
                    <div class="{label_class}">{result['emoji']} {result['label']}</div>
                    <div class="result-sub">
                        Confidence: <strong>{result['confidence']*100:.1f}%</strong>
                        &nbsp;|&nbsp; Analyzed in {result['inference_ms']} ms
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # --- Gauge chart ---
                st.plotly_chart(
                    make_gauge_chart(result["probability"], result["label"], result["color"]),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                if show_raw_prob:
                    c1, c2 = st.columns(2)
                    c1.metric("Fake Probability", f"{result['probability']*100:.2f}%")
                    c2.metric("Real Probability", f"{(1-result['probability'])*100:.2f}%")

        # --- GradCAM ---
        if show_gradcam and predictor and result.get("gradcam_pil"):
            st.divider()
            st.markdown('<div class="section-title">🔬 GradCAM Explainability — Where the model looked</div>',
                        unsafe_allow_html=True)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(
                    Image.fromarray(result["face_np"]),
                    caption         = "Detected Face (Cropped)",
                    use_column_width = True,
                )
            with gc2:
                st.image(
                    result["gradcam_pil"],
                    caption         = "GradCAM Heatmap — Red = High Suspicion",
                    use_column_width = True,
                )
            st.caption(
                "🔴 **Red/warm regions** indicate areas that contributed most to the "
                "FAKE prediction. Common locations: eye boundaries, jawline, skin texture."
            )

    # ----------------------------------------------------------------
    # VIDEO PREDICTION
    # ----------------------------------------------------------------
    elif file_type == "video":
        st.markdown('<div class="section-title">🎬 Video Analysis</div>', unsafe_allow_html=True)

        # Save temp video
        with tempfile.NamedTemporaryFile(
            suffix  = Path(uploaded.name).suffix,
            delete  = False,
            dir     = "." ,
        ) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        if predictor is None:
            st.info("🎓 Demo mode — train a model to see predictions.")
        else:
            with st.spinner(f"🧠 Analyzing video — sampling up to {max_frames} frames..."):
                try:
                    result = predictor.predict_video(
                        video_path    = tmp_path,
                        max_frames    = max_frames,
                        sample_rate   = 10,
                        return_gradcam= show_gradcam,
                    )
                finally:
                    import os as _os
                    _os.unlink(tmp_path)

            # Summary card
            label_class = "result-label-fake" if result["is_fake"] else "result-label-real"
            st.markdown(f"""
            <div class="result-card">
                <div class="{label_class}">{result['emoji']} VIDEO IS {result['label']}</div>
                <div class="result-sub">
                    Confidence: <strong>{result['confidence']*100:.1f}%</strong>
                    &nbsp;|&nbsp; {result['frames_analyzed']} frames analyzed
                    &nbsp;|&nbsp; {result['inference_ms']/1000:.1f}s
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-frame chart
            st.plotly_chart(
                make_frame_chart(result["frame_probs"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            # Stats
            probs = result["frame_probs"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Fake Prob", f"{np.mean(probs)*100:.1f}%")
            c2.metric("Max Fake Prob",  f"{np.max(probs)*100:.1f}%")
            c3.metric("Min Fake Prob",  f"{np.min(probs)*100:.1f}%")
            c4.metric("Fake Frames",    f"{sum(1 for p in probs if p>=0.5)}/{len(probs)}")

            # GradCAM for most suspicious frame
            if show_gradcam and result.get("gradcam_pil"):
                st.divider()
                st.markdown('<div class="section-title">🔬 Most Suspicious Frame — GradCAM</div>',
                            unsafe_allow_html=True)
                st.image(result["gradcam_pil"], caption="GradCAM on most suspicious frame",
                         use_column_width=False, width=400)

# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.divider()
col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    st.caption("🤖 Model: EfficientNet-B4")
with col_f2:
    st.caption(f"📊 Sessions analyzed: {st.session_state.analysis_count}")
with col_f3:
    st.caption("⚡ Built with PyTorch + Streamlit")
