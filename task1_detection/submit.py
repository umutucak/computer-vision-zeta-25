#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import functional as F
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

ID_TO_CLASS_NAME = {
    1:"VenusExpress", 2:"Cheops", 3:"LisaPathfinder", 4:"ObservationSat1",
    5:"Proba2", 6:"Proba3", 7:"Proba3ocs", 8:"Smart1", 9:"Soho", 10:"XMM Newton"
}

def create_detector(num_classes=11):
    model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def clamp_box_xyxy(box, W=1024, H=1024):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), W - 1))
    x2 = max(0.0, min(float(x2), W - 1))
    y1 = max(0.0, min(float(y1), H - 1))
    y2 = max(0.0, min(float(y2), H - 1))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return [x1, y1, x2, y2]

@torch.no_grad()
def predict_top1(model, img_tensor, device, score_thr=0.0):
    out = model([img_tensor.to(device)])[0]
    if len(out["boxes"]) == 0:
        return None, None, 0.0
    boxes = out["boxes"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()

    keep = scores >= score_thr
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    if boxes.shape[0] == 0:
        return None, None, 0.0

    j = int(np.argmax(scores))
    return boxes[j].tolist(), int(labels[j]), float(scores[j])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-csv", default="detection.csv")
    ap.add_argument("--score-thr", type=float, default=0.0)
    ap.add_argument("--no-fallback", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_detector().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    model.eval()

    test_dir = Path(args.test_dir)
    img_files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]])

    rows = []
    for p in tqdm(img_files, desc="Predict test"):
        img = F.to_tensor(Image.open(p).convert("RGB"))
        box, cls, _ = predict_top1(model, img, device, score_thr=args.score_thr)

        if box is None:
            if args.no_fallback:
                continue
            box = [0, 0, 1, 1]
            cls_name = "VenusExpress"
        else:
            box = clamp_box_xyxy(box, W=img.shape[2], H=img.shape[1])
            cls_name = ID_TO_CLASS_NAME.get(cls, "VenusExpress")

        x1, y1, x2, y2 = [int(round(v)) for v in box]
        rows.append({"filename": p.name, "class": cls_name, "bbox": f"({x1}, {y1}, {x2}, {y2})"})

    df = pd.DataFrame(rows, columns=["filename","class","bbox"]).sort_values("filename").reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} rows={len(df)}")

if __name__ == "__main__":
    main()
