"""
Lightweight CNN baseline for shadow detection.

This baseline trains a small encoder-decoder CNN on the same train/hold-out
split as the main pipeline. It does NOT reuse region features or texton
processing; instead it operates directly on resized RGB images.

The implementation keeps dataset handling consistent (same indices), uses
pixel-level BCE loss, and saves its own weights without overwriting other
methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


class RegionConvNet(nn.Module):
    """Lightweight CNN classifier for region patches."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = feats.flatten(1)
        return self.classifier(feats)


class RegionPatchDataset(Dataset):
    """Dataset for region patches and labels."""

    def __init__(self, patches: List[np.ndarray], labels: List[int], size: int = 64):
        self.patches = patches
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size))
        ])

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        img = self.patches[idx]
        lbl = self.labels[idx]
        img_t = self.transform(img)
        return img_t, torch.tensor(lbl, dtype=torch.float32)


def _compute_region_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute region-level BER/Accuracy/FPR/FNR."""
    preds = preds.astype(int)
    labels = labels.astype(int)
    tp = np.logical_and(preds == 1, labels == 1).sum()
    fp = np.logical_and(preds == 1, labels == 0).sum()
    fn = np.logical_and(preds == 0, labels == 1).sum()
    tn = np.logical_and(preds == 0, labels == 0).sum()
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    ber = 0.5 * (fpr + fnr)
    return {"accuracy": float(acc), "ber": float(ber), "fpr": float(fpr), "fnr": float(fnr)}


def _compute_pixel_metrics_from_stats(
    region_preds: np.ndarray,
    region_pixel_stats: np.ndarray
) -> Dict[str, float]:
    """Compute pixel-level metrics using region pixel stats (n_pixels, n_shadow_pixels)."""
    region_preds = region_preds.astype(np.int32).ravel()
    stats = np.asarray(region_pixel_stats, dtype=np.int64)
    n_pixels = stats[:, 0].astype(np.float64)
    n_shadow = stats[:, 1].astype(np.float64)
    n_non_shadow = n_pixels - n_shadow
    pred_shadow = region_preds == 1
    tp = np.sum(n_shadow[pred_shadow])
    fp = np.sum(n_non_shadow[pred_shadow])
    fn = np.sum(n_shadow[~pred_shadow])
    tn = np.sum(n_non_shadow[~pred_shadow])
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    ber = 0.5 * (fpr + fnr)
    return {"accuracy": float(acc), "ber": float(ber), "fpr": float(fpr), "fnr": float(fnr)}


def run_cnn_region_baseline(
    train_patches: List[np.ndarray],
    train_labels: List[int],
    holdout_patches: List[np.ndarray],
    holdout_labels: List[int],
    holdout_region_stats: List[List[int]],
    config: Dict,
    device: torch.device,
    log_fn,
) -> Dict[str, float]:
    """
    Train and evaluate a region-level CNN baseline.
    """
    resize_size = config.get("cnn_resize", 64)
    epochs = config.get("cnn_epochs", 5)
    batch_size = config.get("cnn_batch_size", 64)
    lr = config.get("cnn_lr", 1e-3)
    num_workers = config.get("cnn_num_workers", 0)

    model = RegionConvNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_set = RegionPatchDataset(train_patches, train_labels, size=resize_size)
    holdout_set = RegionPatchDataset(holdout_patches, holdout_labels, size=resize_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    holdout_loader = DataLoader(holdout_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    log_fn(f"[CNN-Region] Training on {len(train_set)} regions, hold-out {len(holdout_set)} regions")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        log_fn(f"[CNN-Region] Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # Evaluation on hold-out
    model.eval()
    all_preds = []
    with torch.no_grad():
        for imgs, _ in holdout_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            batch_preds = (probs > 0.5).astype(int)
            all_preds.append(batch_preds)
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
    else:
        all_preds = np.array([], dtype=int)

    region_metrics = _compute_region_metrics(all_preds, np.array(holdout_labels))
    pixel_metrics = _compute_pixel_metrics_from_stats(all_preds, np.array(holdout_region_stats))

    log_fn("=== CNN Region Baseline Results on Hold-out ===")
    log_fn(f"  Regions evaluated: {len(holdout_labels)}")
    log_fn(f"  Region BER: {region_metrics['ber']*100:.2f}%")
    log_fn(f"  Region Accuracy: {region_metrics['accuracy']*100:.2f}%")
    log_fn(f"  Pixel BER (expanded): {pixel_metrics['ber']*100:.2f}%")
    log_fn(f"  Pixel Accuracy (expanded): {pixel_metrics['accuracy']*100:.2f}%")
    log_fn(f"  Pixel FPR (expanded): {pixel_metrics['fpr']*100:.2f}%")
    log_fn(f"  Pixel FNR (expanded): {pixel_metrics['fnr']*100:.2f}%")

    # Save model without touching other methods
    save_dir = Path(config['output_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "region_metrics": region_metrics,
            "pixel_metrics": pixel_metrics,
        },
        save_dir / "sbu_model_cnn.pt"
    )
    return pixel_metrics

