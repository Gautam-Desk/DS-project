# 🛡️ DeepGuard AI — Deepfake Detection & Forensic Suite

> **State-of-the-Art Deepfake Vision System combining Spatial Neural Attention (EfficientNet-B4 + GradCAM) with Frequency-Domain Forensic Spectrum Analysis (2D FFT).**

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.14-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61+-brightgreen.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Key Capabilities

1. **💬 Conversational Forensic AI Assistant**
   - Natural language chat with DeepGuard AI to discuss deepfake generation methods, explainability, and forensic insights.
2. **📷 Multi-Modal Deepfake Detection**
   - Single portrait and video multi-frame temporal inspection.
   - Real-time **Live Webcam snapshot** detection directly in your browser.
3. **🔬 Dual-Domain Explainability**
   - **Spatial GradCAM Heatmaps**: Identifies suspicious boundary warping, eye/pupil anomalies, and facial blending seams.
   - **Frequency Domain (2D FFT Magnitude Spectrum)**: Exposes periodic high-frequency checkerboard grid patterns characteristic of GAN and Latent Diffusion upsampling layers.
4. **🎬 Video Multi-Frame Timeline**
   - Analyzes temporal frame sequences, flagging peak suspicious timestamps and worst-frame GradCAM attention.
5. **📄 Instant Forensic PDF / Text Report Export**
   - One-click export of analysis logs with cryptographic SHA-256 fingerprint, risk tier, and anomaly breakdown.

---

## 📁 Project Architecture

```
DSproject/
├── app/
│   └── app.py                  ← Streamlit interactive forensic web application
├── src/
│   ├── model.py                ← Torchvision EfficientNet-B4/B0 binary classifier
│   ├── preprocess.py           ← OpenCV face detector & 2D FFT spectral analyzer
│   ├── data_loader.py          ← PyTorch dataset & dataloaders
│   ├── train.py                ← Two-phase training loop with Cosine Annealing
│   ├── evaluate.py             ← ROC curves, confusion matrix, precision-recall
│   ├── gradcam.py              ← High-resolution GradCAM explainability engine
│   ├── predict.py              ← Unified high-level inference engine
│   ├── security.py             ← Magic-byte verification, sanitization, rate limiter
│   └── generate_sample_data.py ← Synthetic benchmark dataset generator
├── notebooks/
│   └── Deepfake_Detection_Training_GPU.ipynb ← 1-click Colab/Kaggle GPU training
├── tests/
│   └── test_pipeline.py        ← Automated end-to-end test suite
├── config.yaml                 ← Hyperparameters, paths & thresholds
├── requirements.txt            ← Clean Python package dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Gautam-Desk/DS-project.git
cd DS-project

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Dataset & Train Locally

```bash
# Generate sample benchmark dataset
python -m src.generate_sample_data

# Train the model
python -m src.train

# Evaluate on test split
python -m src.evaluate
```

### 3. Run Automated Tests

```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch the Interactive Web Application

```bash
streamlit run app/app.py
```

Open your browser at **`http://localhost:8501`**

---

## ☁️ Train on Free Cloud GPUs (Google Colab & Kaggle)

For training on large datasets (Celeb-DF, FaceForensics++, DFDC):
1. Open [`notebooks/Deepfake_Detection_Training_GPU.ipynb`](notebooks/Deepfake_Detection_Training_GPU.ipynb) in Google Colab or Kaggle.
2. Enable GPU hardware acceleration (**T4 / V100 / A100**).
3. Run all cells to fine-tune the model and export `models/best_model.pth`.

---

## 🔒 Built-in Security Architecture

| Security Feature | Implementation Details |
|---|---|
| **Magic Byte Verification** | Inspects binary file signatures (JPEG, PNG, WebP, MP4, MOV) to prevent MIME spoofing. |
| **Path Traversal Shield** | Regex sanitization strips relative directory sequences (`../`) and null bytes. |
| **Session Rate Limiter** | Sliding window rate limiting prevents resource exhaustion. |
| **Memory Isolation** | Safe cleanup of temporary video chunks and hook de-allocation. |

---

## 📄 License & Disclaimer

This project is licensed under the **MIT License**.  
*Disclaimer: This software is developed for research, forensic validation, and educational purposes.*
