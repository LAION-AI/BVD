from transformers import AutoProcessor, AutoModel
import torch
from typing import List


class ClipScoreProcessor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def __call__(self, text: List[str], images: List) -> List[float]:
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            print(probs.shape)
        probs = probs.cpu().detach().numpy()
        # take he diagonal of the matrix
        probs = probs.diagonal()
        print(probs.shape)
        return probs
        # inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True).to(self.device)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        #     logits_per_image = outputs.logits_per_image
        #     probs = logits_per_image.softmax(dim=1)
        # return float(outputs.logits.cpu().detach().numpy())
