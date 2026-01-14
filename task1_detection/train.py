#!/usr/bin/env python3
import os
from ast import literal_eval
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# ----------------------------
# Classes
# ----------------------------
CLASS_NAME_TO_ID = {
    "VenusExpress": 1, "Cheops": 2, "LisaPathfinder": 3, "ObservationSat1": 4,
    "Proba2": 5, "Proba3": 6, "Proba3ocs": 7, "Smart1": 8, "Soho": 9, "XMM Newton": 10
}

THRESHOLDS = np.arange(0.50, 0.96, 0.05)

# ----------------------------
# Dataset
# ----------------------------
class SparkTask1Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir, split):
        self.df = pd.read_csv(csv_path)
        self.root = str(root_dir)
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sat = str(row["Class"])
        img_name = str(row["Image name"])
        bbox = literal_eval(row["Bounding box"])

        img_path = os.path.join(self.root, "images", sat, self.split, img_name)
        pil = Image.open(img_path).convert("RGB")
        image = F.to_tensor(pil)

        x1, y1, x2, y2 = bbox
        boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)
        labels = torch.tensor([CLASS_NAME_TO_ID[sat]], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
            "area": torch.tensor([(x2 - x1) * (y2 - y1)], dtype=torch.float32),
        }
        return image, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# ----------------------------
# Model
# ----------------------------
def create_detector(num_classes=11, pretrained=True):
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ----------------------------
# Metric
# ----------------------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

@torch.no_grad()
def evaluate_val_leaderboard(model, val_loader, device, score_thr=0.0):
    model.eval()
    total, misses, score_sum = 0, 0, 0.0

    for images, targets in tqdm(val_loader, desc="Val (leaderboard)"):
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            gt_box = tgt["boxes"][0].cpu().numpy().tolist()
            gt_cls = int(tgt["labels"][0].item())

            if len(out["boxes"]) == 0:
                misses += 1; total += 1; continue

            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()

            keep = scores >= score_thr
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            if boxes.shape[0] == 0:
                misses += 1; total += 1; continue

            j = int(scores.argmax())
            pred_box = boxes[j].tolist()
            pred_cls = int(labels[j])

            iou = iou_xyxy(pred_box, gt_box)
            ok = (pred_cls == gt_cls)
            score_sum += float(np.mean([(iou >= t) and ok for t in THRESHOLDS]))
            total += 1

    return {
        "final50_95": score_sum / max(1, total),
        "miss_rate": misses / max(1, total),
        "n": total
    }

# ----------------------------
# Train
# ----------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    model = create_detector(num_classes=11, pretrained=args.pretrained).to(device)

    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device), strict=True)
        print(f"Resumed from {args.resume}")

    train_ds = SparkTask1Dataset(args.train_csv, args.data_root, split="train")
    val_ds   = SparkTask1Dataset(args.val_csv,   args.data_root, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best = -1.0
    best_path = os.path.join(args.ckpt_dir, "best_by_leaderboard_task1.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())

        print(f"Epoch {epoch} train_loss={loss_sum/len(train_loader):.4f}")

        stats = evaluate_val_leaderboard(model, val_loader, device, score_thr=args.score_thr)
        print(f"Epoch {epoch} VAL final50-95={stats['final50_95']:.4f} miss_rate={stats['miss_rate']:.4f}")

        torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"epoch_{epoch}.pth"))
        if stats["final50_95"] > best:
            best = stats["final50_95"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST -> {best_path}")

    print("Done.")
    print(f"BEST_WEIGHTS={best_path}")

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="root folder containing train.csv/val.csv/images/")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--ckpt-dir", required=True)

    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--score-thr", type=float, default=0.0)
    return p

def main():
    args = build_parser().parse_args()
    # default CSV paths inside data-root
    if args.train_csv is None:
        args.train_csv = str(Path(args.data_root) / "train.csv")
    if args.val_csv is None:
        args.val_csv = str(Path(args.data_root) / "val.csv")
    train(args)

if __name__ == "__main__":
    main()
