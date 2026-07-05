import torch

DEPTH = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768
IMAGE_ENCODER_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
IMAGE_SIZE = 384
MAX_TEXT_TOKENS = 128
MODEL_PATH = "../models/weights.pt"
NUM_HEADS = 12
TEXT_ENCODER_ID = "intfloat/e5-base-v2"
