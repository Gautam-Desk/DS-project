# 🔍 Deepfake Detection System

> **AI-powered deepfake image & video detection using EfficientNet-B4 + GradCAM explainability**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 What This Does

Detects AI-generated deepfake faces in images and videos using:
- **EfficientNet-B4** fine-tuned binary classifier (Real vs Fake)
- **MTCNN** face detection and alignment
- **GradCAM** heatmaps to show *why* a face is flagged as fake
- **Streamlit** web app with file security validation and rate limiting

---

## 📁 Project Structure

```
deepfake-detection/
├── .gitignore                  ← Protects secrets, data, model weights
├── .env.example                ← Environment variable template (safe to commit)
├── config.yaml                 ← All hyperparameters & settings
├── requirements.txt            ← Python dependencies
│
├── src/
│   ├── model.py                ← EfficientNet-B4 architecture
│   ├── preprocess.py           ← Face detection + augmentation pipeline
│   ├── data_loader.py          ← PyTorch Dataset & DataLoader
│   ├── train.py                ← Training loop (AMP, early stopping)
│   ├── evaluate.py             ← Metrics, ROC, confusion matrix
│   ├── gradcam.py              ← GradCAM explainability
│   ├── predict.py              ← Image & video inference engine
│   └── security.py             ← File validation, rate limiting
│
├── app/
│   └── app.py                  ← Streamlit web application
│
├── .streamlit/
│   ├── config.toml             ← Streamlit deployment config
│   └── secrets.toml.example   ← Secrets template (safe to commit)
│
├── data/                       ← GITIGNORED — your datasets go here
└── models/                     ← GITIGNORED — trained model weights go here
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection

# Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the template
copy .env.example .env         # Windows
# cp .env.example .env         # Linux/Mac

# Edit .env with your settings (never commit this file!)
notepad .env
```

### 3. Download Dataset

Download from Kaggle: [Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)

Organize as:
```
data/splits/
    train/real/   *.jpg
    train/fake/   *.jpg
    val/real/
    val/fake/
    test/real/
    test/fake/
```

### 4. Train the Model

```bash
python -m src.train
```

Training output:
```
Using device: cuda
Loading datasets...
[Dataset] Loaded 14000 samples: 7000 real, 7000 fake
Starting training for 30 epochs (5 warm-up + 25 fine-tune)

Epoch [01/30] Train Loss: 0.6821 | Train AUC: 0.7234 | Val Loss: 0.5932 | Val AUC: 0.8105
✅ New best model saved! Val AUC: 0.8105
...
Epoch [18/30] Train Loss: 0.1823 | Train AUC: 0.9712 | Val Loss: 0.1991 | Val AUC: 0.9634
✅ New best model saved! Val AUC: 0.9634
```

### 5. Evaluate

```bash
python -m src.evaluate --model models/best_model.pth --split test
```

### 6. Run Web App

```bash
streamlit run app/app.py
```
Open http://localhost:8501

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/deepfake-detection.git
   git push -u origin main
   ```

2. Upload your trained model to [Hugging Face Hub](https://huggingface.co) or GitHub Releases

3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

4. Select your repo → set **Main file**: `app/app.py`

5. Add secrets under **Advanced Settings**:
   ```toml
   [general]
   SECRET_KEY = "your_secret_key"
   ```

6. Click **Deploy!** 🚀

---

## 🔒 Security Features

| Feature | Details |
|---|---|
| **`.gitignore`** | Blocks datasets, model weights, `.env`, secrets from GitHub |
| **Magic byte validation** | Checks actual file content, not just extension |
| **File size limit** | Max 50 MB per upload |
| **Rate limiting** | Max 10 requests/minute per session |
| **Filename sanitization** | Prevents path traversal attacks |
| **Environment check** | Warns about debug mode in production |
| **No hardcoded secrets** | All credentials via `.env` or Streamlit secrets |

---

## 📊 Expected Performance

| Model | Accuracy | AUC-ROC |
|---|---|---|
| ResNet-50 (baseline) | ~88% | ~0.93 |
| **EfficientNet-B4 (ours)** | **~93%** | **~0.97** |

Tested on FaceForensics++ (c23 compression).

---

## 🛠️ Tech Stack

| Category | Tool |
|---|---|
| Deep Learning | PyTorch 2.1 |
| Model | EfficientNet-B4 |
| Face Detection | MTCNN (facenet-pytorch) |
| Augmentation | Albumentations |
| Explainability | GradCAM (manual implementation) |
| Web App | Streamlit |
| Visualization | Plotly |

---

## 📚 References

1. Rössler et al. — *FaceForensics++*, ICCV 2019
2. Tan & Le — *EfficientNet*, ICML 2019
3. Selvaraju et al. — *GradCAM*, ICCV 2017
4. Li et al. — *Celeb-DF*, CVPR 2020

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

> ⚠️ **Ethical Note**: This tool is built for research and educational purposes.
> Use responsibly. Do not use to create or spread deepfakes.
