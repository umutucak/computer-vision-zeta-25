#!/usr/bin/env python3
"""
Task-2 TRAIN
U-Net (segmentation_models_pytorch) + AMP + cosine warmup + CE+Dice
Leaderboard metric (50-95) with panel discard rule
Fast cv2 IO + optional cached decoded masks
"""

import os, math
from pathlib import Path
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# -------------------------
# Metric config (fixed)
# -------------------------
THRESHOLDS = np.arange(0.50, 0.96, 0.05)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -------------------------
# Helpers: IO / decode / resize
# -------------------------
def read_split_df(root_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(root_dir, f"{split}.csv"))

def bgr_mask_to_labels_fast(mask_bgr: np.ndarray) -> np.ndarray:
    # cv2 loads BGR
    b = mask_bgr[..., 0]
    g = mask_bgr[..., 1]
    r = mask_bgr[..., 2]
    out = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    body   = (r > 200) & (g < 80) & (b < 80)
    panels = (b > 200) & (g < 80) & (r < 80)
    out[body] = 1
    out[panels] = 2
    return out

def resize_img_rgb(img_rgb: np.ndarray, size: int) -> np.ndarray:
    if img_rgb.shape[0] == size and img_rgb.shape[1] == size:
        return img_rgb
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)

def resize_mask_nn(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


# -------------------------
# Dataset
# -------------------------
class SparkTask2Dataset(torch.utils.data.Dataset):
    """
    Returns:
      img: float32 tensor (3,H,W) normalized (ImageNet)
      mask: int64 tensor (H,W) with {0,1,2}
    """
    def __init__(self, root_dir: str, split="train", img_size=1024, cache_dir=None):
        self.root = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        df = read_split_df(str(self.root), split)
        self.sats = df["Class"].astype(str).tolist()
        self.img_names = df["Image name"].astype(str).tolist()
        self.mask_names = df["Mask name"].astype(str).tolist()

    def __len__(self):
        return len(self.img_names)

    def _cache_path(self, sat, img_name):
        key = f"{sat}_{self.split}_{Path(img_name).stem}_{self.img_size}.npz"
        return self.cache_dir / key

    def __getitem__(self, idx):
        sat = self.sats[idx]
        img_name = self.img_names[idx]
        mask_name = self.mask_names[idx]

        img_path = self.root / "images" / sat / self.split / img_name
        mask_path = self.root / "mask" / sat / self.split / mask_name

        # ---- mask labels (cached) ----
        if self.cache_dir:
            cp = self._cache_path(sat, img_name)
            if cp.exists():
                labels = np.load(cp)["labels"].astype(np.uint8)
            else:
                m = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
                if m is None:
                    raise FileNotFoundError(str(mask_path))
                labels = bgr_mask_to_labels_fast(m)
                labels = resize_mask_nn(labels, self.img_size)
                np.savez_compressed(cp, labels=labels)
        else:
            m = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
            if m is None:
                raise FileNotFoundError(str(mask_path))
            labels = bgr_mask_to_labels_fast(m)
            labels = resize_mask_nn(labels, self.img_size)

        # ---- image ----
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = resize_img_rgb(img_rgb, self.img_size)

        x = img_rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        img_t = torch.from_numpy(x).permute(2, 0, 1).contiguous()
        mask_t = torch.from_numpy(labels.astype(np.int64)).contiguous()
        return img_t, mask_t


# -------------------------
# Loss: CE + Dice
# -------------------------
class CEPlusDice(nn.Module):
    def __init__(self, num_classes=3, ce_w=1.0, dice_w=1.0, label_smoothing=0.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.dice = smp.losses.DiceLoss(mode="multiclass", classes=list(range(num_classes)))
        self.ce_w = ce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        return self.ce_w * self.ce(logits, target) + self.dice_w * self.dice(logits, target)


# -------------------------
# Leaderboard metric
# -------------------------
def iou_bool(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return np.nan
    return float(inter / (union + 1e-9))

def score_from_ious(ious: np.ndarray) -> float:
    ok = (ious[:, None] >= THRESHOLDS[None, :]).astype(np.float32)
    return float(ok.mean())

def eval_task2_leaderboard(pred_labels_list, gt_labels_list, panel_min_ratio: float):
    body_ious, pan_ious = [], []
    pan_used = 0

    for pred, gt in zip(pred_labels_list, gt_labels_list):
        gt_body = (gt == 1)
        gt_pan  = (gt == 2)
        pred_body = (pred == 1)
        pred_pan  = (pred == 2)

        body_ious.append(iou_bool(pred_body, gt_body))

        gt_body_px = gt_body.sum()
        gt_pan_px  = gt_pan.sum()
        if gt_pan_px >= panel_min_ratio * max(1, gt_body_px):
            pan_used += 1
            pan_ious.append(iou_bool(pred_pan, gt_pan))

    body_ious = np.array(body_ious, dtype=np.float32)
    pan_ious  = np.array(pan_ious, dtype=np.float32) if len(pan_ious) else np.array([], dtype=np.float32)

    body50_95 = score_from_ious(body_ious)
    pan50_95  = score_from_ious(pan_ious) if len(pan_ious) else 0.0
    final50_95 = 0.5 * (body50_95 + pan50_95)

    return {
        "final50_95": final50_95,
        "body50_95": body50_95,
        "pan50_95": pan50_95,
        "body_mIoU": float(np.nanmean(body_ious)),
        "pan_mIoU": float(np.nanmean(pan_ious)) if len(pan_ious) else 0.0,
        "panel_imgs": f"{pan_used}/{len(pred_labels_list)}",
    }


# -------------------------
# TTA for val (logit averaging)
# -------------------------
@torch.no_grad()
def tta_logits(model, x):
    logits0 = model(x)

    x_h = torch.flip(x, dims=[3])
    logits_h = torch.flip(model(x_h), dims=[3])

    x_r = torch.rot90(x, k=2, dims=[2, 3])
    logits_r = torch.rot90(model(x_r), k=2, dims=[2, 3])

    return (logits0 + logits_h + logits_r) / 3.0


# -------------------------
# Model / Scheduler
# -------------------------
def make_model(encoder: str, encoder_weights: str, num_classes: int):
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )

def cosine_warmup_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# -------------------------
# Train / Val loops
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, use_amp, epoch, epochs):
    model.train()
    loss_sum = 0.0

    pbar = tqdm(loader, desc=f"Train {epoch:03d}/{epochs}", leave=True)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        scheduler.step()
        loss_sum += float(loss.item())
        pbar.set_postfix(loss=float(loss.item()), lr=float(optimizer.param_groups[0]["lr"]))

    return loss_sum / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, use_tta_val, panel_min_ratio):
    model.eval()
    loss_sum = 0.0
    preds_all, gts_all = [], []

    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = tta_logits(model, imgs) if use_tta_val else model(imgs)
                loss = criterion(logits, masks)
        else:
            logits = tta_logits(model, imgs) if use_tta_val else model(imgs)
            loss = criterion(logits, masks)

        loss_sum += float(loss.item())

        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
        gt = masks.cpu().numpy().astype(np.uint8)
        preds_all.extend(list(pred))
        gts_all.extend(list(gt))

    metrics = eval_task2_leaderboard(preds_all, gts_all, panel_min_ratio=panel_min_ratio)
    return loss_sum / max(1, len(loader)), metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder containing train.csv/val.csv/images/mask")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--cache-dir", default="", help="'' disables; recommend node-local path")

    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--encoder", default="timm-efficientnet-b3")
    ap.add_argument("--encoder-weights", default="imagenet")
    ap.add_argument("--num-classes", type=int, default=3)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)

    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-tta-val", action="store_true")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--panel-min-ratio", type=float, default=0.05)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp)
    use_tta_val = (not args.no_tta_val)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_path = os.path.join(args.ckpt_dir, "best_task2_by_final50_95.pth")

    cache_dir = args.cache_dir.strip()
    if cache_dir == "":
        cache_dir = None

    # speed knobs
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    train_ds = SparkTask2Dataset(args.data_root, "train", img_size=args.img_size, cache_dir=cache_dir)
    val_ds   = SparkTask2Dataset(args.data_root, "val",   img_size=args.img_size, cache_dir=cache_dir)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=True
    )

    model = make_model(args.encoder, args.encoder_weights, args.num_classes).to(device)
    criterion = CEPlusDice(num_classes=args.num_classes, ce_w=1.0, dice_w=1.0, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_final = -1.0
    print(f"Device={device} | Train={len(train_ds)} | Val={len(val_ds)} | steps/epoch={len(train_loader)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device=device, use_amp=use_amp, epoch=epoch, epochs=args.epochs
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion,
            device=device, use_amp=use_amp,
            use_tta_val=use_tta_val,
            panel_min_ratio=args.panel_min_ratio,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"final50-95={val_metrics['final50_95']:.4f} | "
            f"body50-95={val_metrics['body50_95']:.4f} | "
            f"pan50-95={val_metrics['pan50_95']:.4f} | "
            f"body_mIoU={val_metrics['body_mIoU']:.4f} | "
            f"pan_mIoU={val_metrics['pan_mIoU']:.4f} | "
            f"panel_imgs={val_metrics['panel_imgs']}"
        )

        if val_metrics["final50_95"] > best_final:
            best_final = val_metrics["final50_95"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "final50_95": best_final},
                best_path
            )
            print(f"Saved best checkpoint -> {best_path} (final50-95={best_final:.4f})")

    print("Done.")
    print(f"BEST_WEIGHTS={best_path}")

if __name__ == "__main__":
    main()
