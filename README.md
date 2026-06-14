# LAION Big Video Dataset

**A 10-Million-Hour Open Video Dataset for Multimodal Pre-training**

Project Page (coming soon) · Paper (coming soon) · Download (coming soon) · [GitHub](https://github.com/laion-ai/bvd)


## Abstract

We present **LAION-BVD** (LAION - Big Video Dataset), a large-scale open video dataset for multimodal learning, containing *1.3B platform-specific video URLs* collected from CommonCrawl. From these, we download 80M videos with a total duration of *10 million hours*. The dataset is designed for multimodal pre-training across video, audio, and image modalities.

Using content-aware scene detection we extract clips for which we synthetically generate video and audio captions. Models trained on these data achieve competitive performance on standard video-text and audio-text benchmarks, with consistent improvements as training or model scale increases. Additionally, we explore video frames as a new source of image-text data by extracting scene-changing frames. These frames exhibit a visual distribution distinct from standard web image corpora, and models trained on this dataset achieve strong image-text retrieval performance.

Overall, LAION-BVD significantly expands open access to multimodal videos at unprecedented scale and is released to the research community for multimodal research.

## Scale

| Metric | Value |
|---|---|
| Video URLs | 1.3B |
| Downloaded Videos | 80M |
| Total Duration | 10M hours |
| Annotated Clips | 55M |
| Extracted Frames | 300M |

## Results

**Video-Language (ViCLIP):** ViCLIP models trained on LAION-BVD match or exceed InternVid-trained models by up to 4.4% on standard video-text benchmarks, with consistent improvements as training scale grows from 10M to 50M clips.

**Audio-Language (CLAP):** CLAP models trained on LAION-BVD achieve competitive performance against other large-scale uncurated audio datasets, leveraging rich in-the-wild soundscapes extracted directly from video.

**Image-Text (CLIP):** Frame-based CLIP models achieve strong image-text retrieval performance on standard benchmarks. Video frames exhibit a visual distribution distinct from typical web corpora, complementing existing image pre-training sources.

## Pipeline

The full pipeline covers the workflow from raw data to training-ready format:

1. **Download** (`0_download/`) — bulk video acquisition from CommonCrawl URLs
2. **Curation** (`1_curation/`) — scene detection and splitting, frame extraction
3. **Captioning** (`2_captioning/`) — audio, frame, and video captioning via VLMs
4. **Training** (`3_training/`) — training of ViCLIP, CLAP, and CLIP

## Ethics & Release Statement

LAION-BVD is released to support open and reproducible multimodal research at scale. Large-scale video datasets and the models trained on them are increasingly concentrated within a small number of predominantly US-based technology companies, limiting independent scientific investigation and reproducibility. By providing an open resource for academic research, we aim to broaden access to multimodal training data and enable more transparent evaluation of large-scale video, audio, and image models.

LAION-BVD is released **exclusively for research purposes and not for commercial use**. The dataset is intended to support scientific research, reproducibility, safety analysis, and the study of multimodal foundation models and related systems. We encourage users to respect the rights and copyright of content creators and to use the dataset responsibly and in accordance with applicable laws and platform terms.

Like other large-scale web datasets, LAION-BVD may contain biases, stereotypes, and uneven representation across languages, regions, and topics. Models trained on this data may inherit such biases. Researchers using the dataset should be aware of these limitations and, where relevant, evaluate and report them alongside model capabilities.
