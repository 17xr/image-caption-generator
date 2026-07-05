import torch
from io import BytesIO
from torchvision.transforms import InterpolationMode, v2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException

from app.dependencies import generate_captions
from app.config import IMAGE_SIZE

app = FastAPI(
    title="Image Caption Generator",
    description="Caption an image in multiple styles using a modern transformer-based model",
    version="1.0",
)

transform = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop((384, 384)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok"}


@app.post("/caption", tags=["Caption"])
async def caption(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        results = generate_captions(tensor)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
