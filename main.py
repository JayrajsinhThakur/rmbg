import os
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# These must exist in your project root (copied from DIS/IS-Net/)
from data_loader_cache import normalize, im_reader, im_preprocess
from models import ISNetDIS

# Optional: auto-download weights from Hugging Face if not present
# (recommended for Render so you don't commit a big .pth)
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None


# ----------------------------
# Preprocessing
# ----------------------------
class GOSNormalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return normalize(image, self.mean, self.std)


TRANSFORM = transforms.Compose([GOSNormalize()])


def load_image_tensor(image_path: str, cache_size=(1024, 1024)):
    """
    Returns:
      image_tensor: (1,3,H,W)
      orig_hw: (H,W)
    """
    im = im_reader(image_path)                 # numpy
    im, im_shp = im_preprocess(im, cache_size) # resized + original shape
    im = torch.divide(im, 255.0)

    orig_hw = (int(im_shp[0]), int(im_shp[1]))
    image_tensor = TRANSFORM(im).unsqueeze(0)
    return image_tensor, orig_hw


def build_model(model_path: str, device: str):
    net = ISNetDIS()
    state = torch.load(model_path, map_location=device)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


@torch.no_grad()
def predict_mask(net, image_tensor, orig_hw, device: str):
    """
    Returns:
      mask_uint8: (H,W) uint8 [0..255]
    """
    image_tensor = image_tensor.type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False).to(device)

    ds = net(inputs)[0]
    pred = ds[0][0, 0, :, :]

    pred = F.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=orig_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    ma = torch.max(pred)
    mi = torch.min(pred)
    pred = (pred - mi) / (ma - mi + 1e-8)

    mask = (pred.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return mask


def ensure_weights(local_path: str) -> str:
    """
    Ensures the model weights exist.
    - If file exists locally, uses it.
    - If missing and HF download is enabled, downloads isnet-general-use.pth
    """
    if os.path.exists(local_path):
        return local_path

    allow_download = os.getenv("ALLOW_MODEL_DOWNLOAD", "1") == "1"
    if not allow_download:
        raise FileNotFoundError(
            f"Model not found at {local_path} and ALLOW_MODEL_DOWNLOAD=0"
        )

    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub not installed, and model file is missing. "
            "Either install huggingface_hub or provide the .pth in the repo."
        )

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id="NimaBoscarino/IS-Net_DIS-general-use",
        filename="isnet-general-use.pth",
        local_dir=str(Path(local_path).parent),
        local_dir_use_symlinks=False,
    )
    return downloaded


# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(title="IS-Net Background Removal API (Base64 Output)")

# If you call this API from browser frontends, CORS helps:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "1024"))
MODEL_PATH = os.getenv("MODEL_PATH", "saved_models/isnet-general-use.pth")

_net: Optional[ISNetDIS] = None


@app.on_event("startup")
def startup():
    global _net
    weights_path = ensure_weights(MODEL_PATH)
    _net = build_model(weights_path, device=DEVICE)
    print(f"[STARTUP] device={DEVICE} cache_size={CACHE_SIZE} model={weights_path}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "device": DEVICE}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    """
    Input: multipart/form-data with field 'file'
    Output: JSON { image_base64: "...", mime_type: "image/png", filename: "..." }
    """
    if _net is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")

    # DIS helpers expect a file path -> write to temp file
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        suffix = ".png"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        image_tensor, (orig_h, orig_w) = load_image_tensor(
            tmp.name, cache_size=(CACHE_SIZE, CACHE_SIZE)
        )
        mask = predict_mask(_net, image_tensor, (orig_h, orig_w), device=DEVICE)

    # Create RGBA PNG in memory
    img_rgb = Image.open(BytesIO(data)).convert("RGB")
    pil_mask = Image.fromarray(mask).convert("L")

    rgba = img_rgb.copy()
    rgba.putalpha(pil_mask)

    out = BytesIO()
    rgba.save(out, format="PNG")
    png_bytes = out.getvalue()

    b64 = base64.b64encode(png_bytes).decode("utf-8")

    out_name = (Path(file.filename).stem if file.filename else "cutout") + ".png"
    return {
        "filename": out_name,
        "mime_type": "image/png",
        "image_base64": b64,
    }