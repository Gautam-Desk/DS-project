"""
app/app.py — Streamlit Web Application
========================================
Deepfake Detection System — Interactive Web Demo

Fixed:
  - Lazy imports: torch & ML libs only loaded when model runs
  - Graceful demo mode when model not yet trained
  - No crash on missing dependencies

Run locally:
    streamlit run app/app.py
"""

import os
import io
import sys
import uuid
import logging
import tempfile
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pth")
IMAGE_SIZE  = 380

# -----------------------------------------------------------------------
# Page Config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Detector | AI Vision",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "AI-powered deepfake detection using EfficientNet-B4 + GradCAM"
    }
)

# -----------------------------------------------------------------------
# Custom CSS
# -----------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

.main .block-container { padding: 2rem 3rem; max-width: 1200px; }

.hero-title {
    font-size: 3rem; font-weight: 700; color: #ffffff;
    letter-spacing: -1px; line-height: 1.1; text-align: center;
}
.hero-sub {
    font-size: 1.05rem; color: #a0aec0;
    margin-top: 0.4rem; text-align: center;
}

.result-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px; padding: 1.5rem 2rem;
    text-align: center; margin: 0.5rem 0;
}
.label-fake { font-size: 2.4rem; font-weight: 700; color: #FC5C7D; }
.label-real { font-size: 2.4rem; font-weight: 700; color: #43E97B; }
.label-sub  { font-size: 0.9rem; color: #a0aec0; margin-top: 0.3rem; }

.section-title {
    font-size: 1rem; font-weight: 600; color: #e2e8f0;
    margin-bottom: 0.6rem;
    border-left: 3px solid #667eea; padding-left: 0.75rem;
}

.info-chip {
    display: inline-block;
    background: rgba(67,233,123,0.15); color: #43E97B;
    border: 1px solid rgba(67,233,123,0.3); border-radius: 20px;
    padding: 0.25rem 0.8rem; font-size: 0.8rem; font-weight: 500; margin: 0.2rem;
}
.warn-chip {
    background: rgba(252,92,125,0.15); color: #FC5C7D;
    border-color: rgba(252,92,125,0.3);
}

.demo-banner {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4); border-radius: 12px;
    padding: 1.2rem 1.5rem; margin: 1rem 0;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
    transition: opacity 0.2s; width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

.stFileUploader > div {
    background: rgba(255,255,255,0.05) !important;
    border: 2px dashed rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
}

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------
if "session_id"     not in st.session_state: st.session_state.session_id     = str(uuid.uuid4())
if "analysis_count" not in st.session_state: st.session_state.analysis_count = 0
if "predictor"      not in st.session_state: st.session_state.predictor      = None

# -----------------------------------------------------------------------
# Lazy model loader — only imports torch if model file exists
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_predictor_safe(model_path: str):
    """Load ML predictor only if model file exists. Returns None in demo mode."""
    if not Path(model_path).exists():
        return None
    try:
        from src.predict import DeepfakePredictor
        return DeepfakePredictor(model_path=model_path, device="auto",
                                 threshold=0.5, image_size=IMAGE_SIZE)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return None

# -----------------------------------------------------------------------
# Security validator — lightweight, no torch needed
# -----------------------------------------------------------------------
def quick_validate(filename: str, file_bytes: bytes, session_id: str):
    """Fast security check without importing heavy ML libs."""
    ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mov"}
    MAX_MB  = 50

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        return False, f"File type '{ext}' not allowed.", {}

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_MB:
        return False, f"File too large: {size_mb:.1f} MB. Max: {MAX_MB} MB.", {}

    import hashlib
    sha = hashlib.sha256(file_bytes).hexdigest()
    ftype = "video" if ext in {".mp4", ".avi", ".mov"} else "image"

    return True, "✅ File validated.", {
        "sha256"    : sha,
        "size_mb"   : round(size_mb, 2),
        "file_type" : ftype,
        "safe_name" : filename,
    }

# -----------------------------------------------------------------------
# Chart helpers
# -----------------------------------------------------------------------
def gauge_chart(prob: float, color: str):
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = prob * 100,
        number= {"suffix": "%", "font": {"size": 38, "color": "#fff"}},
        title = {"text": "Fake Probability", "font": {"size": 13, "color": "#a0aec0"}},
        gauge = {
            "axis"   : {"range": [0, 100], "tickwidth": 1, "tickcolor": "#444"},
            "bar"    : {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(255,255,255,0.04)",
            "borderwidth": 0,
            "steps"  : [
                {"range": [0,  50], "color": "rgba(67,233,123,0.12)"},
                {"range": [50, 75], "color": "rgba(255,165,0,0.12)"},
                {"range": [75,100], "color": "rgba(252,92,125,0.12)"},
            ],
            "threshold": {"line": {"color": "white", "width": 2},
                          "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=30, r=30, t=40, b=10), height=230,
    )
    return fig


def frame_chart(probs: list):
    colors = ["#FC5C7D" if p >= 0.5 else "#43E97B" for p in probs]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(probs)+1)),
        y=[p*100 for p in probs],
        marker_color=colors,
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="rgba(255,255,255,0.5)",
                  annotation_text="Threshold 50%", annotation_position="top left")
    fig.update_layout(
        title="Per-Frame Fake Probability",
        xaxis_title="Frame", yaxis_title="Fake %", yaxis_range=[0, 110],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#a0aec0"), title_font=dict(color="#fff", size=13),
        height=280, showlegend=False,
    )
    fig.update_xaxes(showgrid=False, color="#555")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="#555")
    return fig

# -----------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem">
    <div class="hero-title">🔍 Deepfake Detector</div>
    <div class="hero-sub">AI-powered face manipulation detection · EfficientNet-B4 + GradCAM</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# Model status banner
# -----------------------------------------------------------------------
model_ready = Path(MODEL_PATH).exists()

if not model_ready:
    st.markdown("""
    <div class="demo-banner">
        <b style="color:#a78bfa">🎓 Demo Mode — Model not yet trained</b><br>
        <span style="color:#cbd5e1; font-size:0.9rem">
        The full UI is active. Upload any image to explore the interface.<br>
        To enable real predictions: train the model and place
        <code>best_model.pth</code> in the <code>models/</code> folder.
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 How to train the model (click to expand)"):
        st.code("""# Option 1 — Train locally (needs GPU)
python -m src.train

# Option 2 — Free GPU on Kaggle (recommended)
# 1. Upload your dataset to Kaggle
# 2. Create a new notebook, enable GPU
# 3. Run: !python -m src.train
# 4. Download models/best_model.pth
# 5. Place it in your project's models/ folder""", language="bash")

st.divider()

# -----------------------------------------------------------------------
# Upload + Options
# -----------------------------------------------------------------------
col_up, col_opt = st.columns([3, 1])

with col_up:
    st.markdown('<div class="section-title">📤 Upload Image or Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov"],
        help="Max 50 MB · JPG, PNG, WebP, MP4, AVI, MOV",
        label_visibility="collapsed",
    )

with col_opt:
    st.markdown('<div class="section-title">⚙️ Options</div>', unsafe_allow_html=True)
    show_gradcam  = st.toggle("GradCAM Heatmap",  value=True)
    show_scores   = st.toggle("Show Raw Scores",   value=False)
    max_frames    = 50
    if uploaded and uploaded.name.lower().endswith((".mp4", ".avi", ".mov")):
        max_frames = st.slider("Max frames", 10, 100, 50, 10)

# -----------------------------------------------------------------------
# PROCESS
# -----------------------------------------------------------------------
if uploaded:
    file_bytes = uploaded.read()

    # Security check
    with st.spinner("🔒 Validating file..."):
        ok, msg, meta = quick_validate(uploaded.name, file_bytes, st.session_state.session_id)

    if not ok:
        st.error(f"**Security check failed:** {msg}")
        st.stop()

    st.markdown(f"""
    <div>
        <span class="info-chip">✅ Validated</span>
        <span class="info-chip">{meta['size_mb']} MB</span>
        <span class="info-chip">{meta['file_type'].upper()}</span>
        <span class="info-chip">SHA: {meta['sha256'][:12]}…</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---- Load predictor (lazy) ----
    predictor = load_predictor_safe(MODEL_PATH)

    # ==================================================================
    # IMAGE
    # ==================================================================
    if meta["file_type"] == "image":
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        c_img, c_res = st.columns([1, 1])

        with c_img:
            st.markdown('<div class="section-title">🖼️ Input</div>', unsafe_allow_html=True)
            st.image(pil_img, use_column_width=True)

        with c_res:
            st.markdown('<div class="section-title">📊 Result</div>', unsafe_allow_html=True)

            if predictor is None:
                # ---- DEMO MODE ----
                import random
                demo_prob  = round(random.uniform(0.05, 0.95), 3)
                is_fake    = demo_prob >= 0.5
                label      = "FAKE" if is_fake else "REAL"
                emoji      = "🚨" if is_fake else "✅"
                color      = "#FC5C7D" if is_fake else "#43E97B"
                cls        = "label-fake" if is_fake else "label-real"
                confidence = demo_prob if is_fake else 1 - demo_prob

                st.markdown(f"""
                <div class="result-card">
                    <div class="{cls}">{emoji} {label} <span style="font-size:0.8rem;color:#888">(demo)</span></div>
                    <div class="label-sub">Confidence: {confidence*100:.1f}% · Demo Mode</div>
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(gauge_chart(demo_prob, color),
                                use_container_width=True, config={"displayModeBar": False})
                st.info("🎓 This is a **random demo result**. Train the model for real predictions.")

            else:
                # ---- REAL PREDICTION ----
                with st.spinner("🧠 Analyzing..."):
                    result = predictor.predict_image(pil_img, return_gradcam=show_gradcam)
                    st.session_state.analysis_count += 1

                cls = "label-fake" if result["is_fake"] else "label-real"
                st.markdown(f"""
                <div class="result-card">
                    <div class="{cls}">{result['emoji']} {result['label']}</div>
                    <div class="label-sub">
                        Confidence: {result['confidence']*100:.1f}%
                        &nbsp;·&nbsp; {result['inference_ms']} ms
                    </div>
                </div>""", unsafe_allow_html=True)

                st.plotly_chart(gauge_chart(result["probability"], result["color"]),
                                use_container_width=True, config={"displayModeBar": False})

                if show_scores:
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Fake Prob", f"{result['probability']*100:.2f}%")
                    sc2.metric("Real Prob", f"{(1-result['probability'])*100:.2f}%")

                # GradCAM
                if show_gradcam and result.get("gradcam_pil"):
                    st.divider()
                    st.markdown('<div class="section-title">🔬 GradCAM — Where the AI looked</div>',
                                unsafe_allow_html=True)
                    g1, g2 = st.columns(2)
                    with g1:
                        st.image(Image.fromarray(result["face_np"]),
                                 caption="Detected Face", use_column_width=True)
                    with g2:
                        st.image(result["gradcam_pil"],
                                 caption="🔴 Red = High suspicion region", use_column_width=True)
                    st.caption("Common fake regions: eye boundaries · jawline · skin texture · mouth corners")

    # ==================================================================
    # VIDEO
    # ==================================================================
    elif meta["file_type"] == "video":
        st.markdown('<div class="section-title">🎬 Video Analysis</div>', unsafe_allow_html=True)

        if predictor is None:
            # Demo mode for video
            import random
            n_frames   = max_frames
            probs      = [round(random.uniform(0.1, 0.9), 3) for _ in range(n_frames)]
            avg_prob   = round(float(np.mean(probs)), 3)
            is_fake    = avg_prob >= 0.5
            label      = "FAKE" if is_fake else "REAL"
            emoji      = "🚨" if is_fake else "✅"
            color      = "#FC5C7D" if is_fake else "#43E97B"
            cls        = "label-fake" if is_fake else "label-real"
            confidence = avg_prob if is_fake else 1 - avg_prob

            st.markdown(f"""
            <div class="result-card">
                <div class="{cls}">{emoji} VIDEO IS {label} <span style="font-size:0.8rem;color:#888">(demo)</span></div>
                <div class="label-sub">Confidence: {confidence*100:.1f}% · {n_frames} frames · Demo Mode</div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(frame_chart(probs), use_container_width=True,
                            config={"displayModeBar": False})

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Mean Fake", f"{np.mean(probs)*100:.1f}%")
            mc2.metric("Max Fake",  f"{np.max(probs)*100:.1f}%")
            mc3.metric("Min Fake",  f"{np.min(probs)*100:.1f}%")
            mc4.metric("Fake Frames", f"{sum(1 for p in probs if p>=0.5)}/{len(probs)}")
            st.info("🎓 These are **random demo values**. Train the model for real predictions.")

        else:
            suffix   = Path(uploaded.name).suffix
            tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_file.write(file_bytes)
            tmp_file.close()

            with st.spinner(f"🧠 Analyzing {max_frames} frames..."):
                try:
                    result = predictor.predict_video(
                        video_path=tmp_file.name,
                        max_frames=max_frames,
                        sample_rate=10,
                        return_gradcam=show_gradcam,
                    )
                finally:
                    os.unlink(tmp_file.name)

            st.session_state.analysis_count += 1
            cls = "label-fake" if result["is_fake"] else "label-real"
            st.markdown(f"""
            <div class="result-card">
                <div class="{cls}">{result['emoji']} VIDEO IS {result['label']}</div>
                <div class="label-sub">
                    Confidence: {result['confidence']*100:.1f}%
                    · {result['frames_analyzed']} frames
                    · {result['inference_ms']/1000:.1f}s
                </div>
            </div>""", unsafe_allow_html=True)

            st.plotly_chart(frame_chart(result["frame_probs"]),
                            use_container_width=True, config={"displayModeBar": False})

            pr = result["frame_probs"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Mean Fake",   f"{np.mean(pr)*100:.1f}%")
            mc2.metric("Max Fake",    f"{np.max(pr)*100:.1f}%")
            mc3.metric("Min Fake",    f"{np.min(pr)*100:.1f}%")
            mc4.metric("Fake Frames", f"{sum(1 for p in pr if p>=0.5)}/{len(pr)}")

            if show_gradcam and result.get("gradcam_pil"):
                st.divider()
                st.markdown('<div class="section-title">🔬 Most Suspicious Frame — GradCAM</div>',
                            unsafe_allow_html=True)
                st.image(result["gradcam_pil"], width=380,
                         caption="GradCAM on most suspicious frame")

# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.divider()
f1, f2, f3 = st.columns(3)
with f1: st.caption("🤖 EfficientNet-B4 + GradCAM")
with f2: st.caption(f"📊 Analyzed this session: {st.session_state.analysis_count}")
with f3: st.caption("⚡ PyTorch + Streamlit")
