def deepseek_vl2(question):
    prompt = f"<|User|>: <image>\n{question}\n\n<|Assistant|>:"
    return prompt


def idefics3(question):
    prompt = f"<|begin_of_text|>User:<image>{question}<end_of_utterance>\nAssistant:"
    return prompt


def internvl(question):
    prompt = f"Image: <image>\nCaption: {question}\n\n"
    return prompt


def qwen2_5_vl(question):
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def paligemma(question):
    return "caption en"


def phi3v(question):
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
    return prompt


PROMPT_MAP = {
    "Qwen/Qwen2.5-VL-3B-Instruct": qwen2_5_vl,
    "BytedanceDouyinContent/SAIL-VL-2B": qwen2_5_vl,
    "OpenGVLab/InternVL2_5-2B-MPO": internvl,
    "OpenGVLab/InternVL2_5-2B": internvl,
    "microsoft/Phi-3.5-vision-instruct": phi3v,
    "deepseek-ai/deepseek-vl2-tiny": deepseek_vl2,
    "HuggingFaceTB/SmolVLM-Instruct": idefics3,
    "google/paligemma2-3b-pt-448": paligemma,
    "OpenGVLab/InternVL2_5-1B": internvl,
}
