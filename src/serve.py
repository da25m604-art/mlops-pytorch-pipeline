import io
import logging
import os
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from dataset import CIFAR10_MEAN, CIFAR10_STD, class_names
from model import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serve")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

STATE = {"model": None, "classes": [], "transform": None, "device": None}


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    config = checkpoint.get("config", {})
    architecture = config.get("model", {}).get("architecture", "resnet18")
    num_classes = config.get("model", {}).get("num_classes", 10)
    dataset = config.get("data", {}).get("dataset", "cifar10")

    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    STATE["model"] = model
    STATE["device"] = device
    STATE["classes"] = class_names(dataset)
    STATE["transform"] = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])
    logger.info(
        "model loaded: arch=%s classes=%d device=%s val_acc=%s",
        architecture, num_classes, device, checkpoint.get("val_accuracy"),
    )


@asynccontextmanager
async def lifespan(app):
    # keep the server up even if loading fails so /health can report 503
    try:
        load_model()
    except Exception:
        logger.exception("failed to load checkpoint at %s", CHECKPOINT_PATH)
    yield


app = FastAPI(title="mlops-pytorch-pipeline serving", lifespan=lifespan)


@app.get("/health")
def health():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    tensor = STATE["transform"](pil).unsqueeze(0).to(STATE["device"])
    with torch.no_grad():
        logits = STATE["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    classes = STATE["classes"]
    top = int(max(range(len(probs)), key=probs.__getitem__))
    return {
        "predicted_class": classes[top],
        "confidence": round(probs[top], 4),
        "probabilities": {c: round(p, 4) for c, p in zip(classes, probs)},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
