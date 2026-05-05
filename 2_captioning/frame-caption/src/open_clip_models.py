import open_clip
from open_clip.tokenizer import _tokenizer
from PIL import Image
import dataclasses
from typing import List, Dict
import os
import numpy as np
import torch


OPEN_CLIP_CACHE_DIR = os.environ.get("OPEN_CLIP_CACHE_DIR", "~/.cache/open_clip_cache")
os.makedirs(OPEN_CLIP_CACHE_DIR, exist_ok=True)


@dataclasses.dataclass
class Output:
    text: str
    score: float


@dataclasses.dataclass
class GenerationOutput:
    outputs: List[Output]


class OpenClipModel:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        model_name, pretrained = model_path.split(":")
        model, _, transform = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=OPEN_CLIP_CACHE_DIR
        )
        self.model = model.to(device)
        self.transform = transform
        self.device = device
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_features(self, images: List[Image.Image]) -> List[np.ndarray]:
        return open_clip.get_image_features(self.model, images)

    def get_text_features(self, texts: List[str]) -> List[np.ndarray]:
        return open_clip.get_text_features(self.model, texts)

    def get_image_text_features(
        self, images: List[Image.Image], texts: List[str]
    ) -> List[np.ndarray]:
        return open_clip.get_image_text_features(self.model, images, texts)

    def pil_to_tensor(self, img: Image.Image) -> np.ndarray:
        return self.transform(img).unsqueeze(0).to(self.device)

    def generate(
        self, inputs: List[Dict[str, str]], sampling_params
    ) -> GenerationOutput:
        images = [i["multi_modal_data"]["image"] for i in inputs]

        if "max_tokens" in sampling_params:
            sampling_params["max_seq_len"] = sampling_params.pop("max_tokens")

        images = [self.transform(img).unsqueeze(0) for img in images]
        images = (
            torch.stack(images).to(self.device).squeeze(1)
        )  # (batch_size, num_channels, height, width)

        out = self.model.generate(images, **sampling_params)
        captions = [
            _tokenizer.decode(i)
            .split("<end_of_text>")[0]
            .replace("<start_of_text>", "")
            .strip()
            for i in out.cpu().numpy()
        ]
        captions = [
            GenerationOutput(outputs=[Output(text=decoded, score=0.0)])
            for decoded in captions
        ]
        return captions
