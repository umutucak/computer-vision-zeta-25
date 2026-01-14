#!/usr/bin/env python3
import os
from pathlib import Path
import argparse

import cv2
import numpy as np
from tqdm import tqdm

import torch
import segmentation_models_pytorch as smp


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def tta_logits(model, x):
    # x: (1,3,H,W)
    l0 = model(x)
    xh = torch.flip(x, dims=[3])
    lh = torch.flip(model(xh), dims=[3])
    xr = torch.rot90(x, k=2, dims=[2, 3])
    lr = torch.rot90(model(xr), k=2, dims=[2, 3])
    return (l0 + lh + lr) / 3.0

def name_to_layer_npz(fn: str) -> str:
    if fn.endswith("_img.jpg"):
        return fn.replace("_img.jpg", "_layer.npz")
    if fn.endswith("_img.png"):
        return fn.replace("_img.png", "_layer.npz")
    return Path(fn).stem + "_layer.npz"

def build_model(encoder: str, num_classes: int, device: str):
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,   # avoid downloads in HPC; weights come from ckpt
        in_channels=3,
        classes=num_classes,
    ).to(device)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True, help="folder with test images (e.g. .../test_seg_images/images)")
    ap.add_argument("--ckpt", required=True, help="path to best_task2_by_final50_95.pth")
    ap.add_argument("--out-dir", required=True, help="folder to write *_layer.npz")
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.out_dir, exist_ok=True)

    model = build_model(args.encoder, args.num_classes, device)
    ckpt = torch.load(args.ckpt, map_location=device)

    # accept either {"model": ...} or raw state_dict
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    test_dir = Path(args.test_dir)
    files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    use_amp = (not args.no_amp) and (device == "cuda")

    for p in tqdm(files, desc="Predict -> NPZ"):
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        H0, W0 = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # resize to model input size
        if (H0, W0) != (args.img_size, args.img_size):
            img_rgb_rs = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
        else:
            img_rgb_rs = img_rgb

        x = img_rgb_rs.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = tta_logits(model, x) if args.tta else model(x)

        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        # output format: bool HxWx3, R=body, B=panels
        out = np.zeros((args.img_size, args.img_size, 3), dtype=np.bool_)
        out[..., 0] = (pred == 1)  # body -> R
        out[..., 2] = (pred == 2)  # panels -> B

        # resize back if original size differs
        if (H0, W0) != (args.img_size, args.img_size):
            out_u8 = out.astype(np.uint8)
            out_u8 = cv2.resize(out_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)
            out = out_u8.astype(np.bool_)

        out_name = name_to_layer_npz(p.name)
        np.savez_compressed(os.path.join(args.out_dir, out_name), data=out)

    print(f"✅ Done. Zip '{args.out_dir}' and submit.")

if __name__ == "__main__":
    main()
