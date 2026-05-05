# LAION-BVD

Pipeline for building large-scale video datasets, covering the full workflow from raw data to training-ready format:

1. **Download** (`0_download/`) - bulk video acquisition
2. **Curation** (`1_curation/`) - scene detection and splitting, frame extraction
3. **Captioning** (`2_captioning/`) - audio, frame, and video captioning via VLMs
4. **Training** (`3_training/`) - training of ViCLIP, CLAP and CLIP