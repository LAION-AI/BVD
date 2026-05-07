# Training Code

This file links the training code used for the multimodal experiments in the paper.

## Video-Text Training (ViCLIP)

For video-language training we integrate ViCLIP into the OpenCLIP codebase:

- [open_clip_video](https://github.com/LAION-AI/open_clip_video?utm_source=chatgpt.com)

For video evaluation and benchmarking we extend CLIP Benchmark with video support:

- [CLIP_benchmark_video](https://github.com/LAION-AI/CLIP_benchmark_video?utm_source=chatgpt.com)

For large-scale experiment management and distributed training runs we use:

- [open_clip_video_autoexp](https://github.com/LAION-AI/open_clip_video_autoexp?utm_source=chatgpt.com)

## Audio-Text Training (CLAP)

For audio-language training we use the LAION-CLAP implementation and training pipeline:

- [LAION-CLAP](https://github.com/LAION-AI/CLAP?utm_source=chatgpt.com)

## Image-Text Training (CLIP)

For image-language training we use the OpenCLIP implementation:

- [OpenCLIP](https://github.com/mlfoundations/open_clip?utm_source=chatgpt.com)
