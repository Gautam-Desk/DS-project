This directory stores trained model checkpoints.

⚠️ MODEL FILES ARE GITIGNORED — .pth files are NOT committed to GitHub.
   (They can be 100+ MB and may encode training data.)

TO SHARE YOUR MODEL:
  Upload to Hugging Face Hub:
    huggingface-cli upload your-username/deepfake-detector models/best_model.pth

  Or use GitHub Releases (for files < 2GB):
    gh release create v1.0 models/best_model.pth

EXPECTED FILES (after training):
models/
  best_model.pth    <- Best checkpoint (by Val AUC)
