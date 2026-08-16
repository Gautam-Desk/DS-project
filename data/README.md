This directory holds raw datasets (FaceForensics++, Kaggle, etc.)

⚠️ DATA IS GITIGNORED — never committed to GitHub.

RECOMMENDED DATASETS:
1. Kaggle - Real and Fake Face Detection (beginner-friendly, ~2k images)
   https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

2. FaceForensics++ (research-grade, requires form submission)
   https://github.com/ondyari/FaceForensics

3. DFDC Dataset (Deepfake Detection Challenge)
   https://www.kaggle.com/competitions/deepfake-detection-challenge

EXPECTED STRUCTURE (after organizing):
data/
  raw/           <- original downloaded files
  processed/     <- extracted + preprocessed faces
  splits/
    train/
      real/      <- real face images
      fake/      <- deepfake images
    val/
      real/
      fake/
    test/
      real/
      fake/
