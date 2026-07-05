import torch
from torch import nn
from app.config import (
    DEPTH,
    DEVICE,
    EMBED_DIM,
    MODEL_PATH,
    NUM_HEADS,
    IMAGE_SIZE,
    MAX_TEXT_TOKENS,
    TEXT_ENCODER_ID,
)

from architecture.transformer import ImageCaption
from transformers import AutoTokenizer
from utils.utils import nucleus_sampling_generate

tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID)
tokenizer.add_special_tokens(
    {
        "additional_special_tokens": [
            "[IMG_START]",
            "[IMG_END]",
            "[DESCRIPTIVE]",
            "[NARRATIVE]",
            "[CONCISE]",
        ]
    }
)

token_embeddings = nn.Embedding(len(tokenizer), EMBED_DIM)
model = ImageCaption(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    depth=DEPTH,
    image_size=IMAGE_SIZE,
    max_text_tokens=MAX_TEXT_TOKENS,
    token_embeddings=token_embeddings,
    tokenizer=tokenizer,
)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE))
model.to(device=DEVICE, dtype=torch.bfloat16)
model.eval()


def generate_captions(input_tensor):
    style_configs = {
        "Concise": {
            "num": 2,
            "temp": 0.20,
            "top_k": 3,
            "top_p": 0.75,
            "rep_penalty": 1.4,
            "ngram": 2,
        },
        "Narrative": {
            "num": 2,
            "temp": 0.75,
            "top_k": 30,
            "top_p": 0.90,
            "rep_penalty": 1.15,
            "ngram": 3,
        },
        "Descriptive": {
            "num": 2,
            "temp": 0.45,
            "top_k": 15,
            "top_p": 0.85,
            "rep_penalty": 1.2,
            "ngram": 3,
        },
    }

    results = nucleus_sampling_generate(
        model=model,
        image_tensor=input_tensor,
        tokenizer=tokenizer,
        device=DEVICE,
        style_configs=style_configs,
        max_length=MAX_TEXT_TOKENS,
    )

    return results
