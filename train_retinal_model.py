"""Train a small PyTorch retinal classifier on correctly labeled image data.

Expected layout:
    datasets/aptos2019/train.csv
    datasets/aptos2019/train_images/<id_code>.png

For DR grading, use diagnosis values 0-4. For diabetes-risk research, use a
verified binary diabetes column (0 = no diabetes, 1 = diabetes). This is for
development and dataset evaluation, not clinical diagnosis.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RetinalDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, image_dir: Path, image_size: int, target_column: str, augment: bool = False):
        self.rows = rows.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
        self.target_column = target_column
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.rows.iloc[index]
        image_path = self.image_dir / f"{row['id_code']}.png"
        if not image_path.exists():
            alternatives = list(self.image_dir.glob(f"{row['id_code']}.*"))
            if not alternatives:
                raise FileNotFoundError(f"Image not found for id_code={row['id_code']}: {image_path}")
            image_path = alternatives[0]
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.fit(image, (self.image_size, self.image_size), method=Image.Resampling.BILINEAR)
        if self.augment and random.random() < 0.5:
            image = ImageOps.mirror(image)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).contiguous()
        return tensor, int(row[self.target_column])


class SmallRetinalCNN(nn.Module):
    def __init__(self, classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 160, 3, padding=1), nn.BatchNorm2d(160), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(160, classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def run_epoch(model, loader, loss_fn, optimizer, device, training: bool):
    model.train(training)
    total_loss = 0.0
    actual: List[int] = []
    predicted: List[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(1).cpu().tolist())
    metrics = {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "quadratic_kappa": cohen_kappa_score(actual, predicted, weights="quadratic"),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a retinal-image classifier on verified labels")
    parser.add_argument("--data-dir", required=True, help="Folder containing train.csv and train_images")
    parser.add_argument("--csv", default="train.csv")
    parser.add_argument("--images", default="train_images")
    parser.add_argument("--output", default="models/aptos_retinal_cnn.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column", default="diagnosis", help="CSV label column; use diabetes for verified binary diabetes labels")
    parser.add_argument("--task", choices=["dr_grading", "diabetes_risk"], default="dr_grading")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    rows = pd.read_csv(data_dir / args.csv)
    required = {"id_code", args.target_column}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    rows[args.target_column] = pd.to_numeric(rows[args.target_column], errors="raise").astype(int)
    if args.task == "diabetes_risk":
        if args.target_column == "diagnosis":
            raise ValueError("Diabetes risk requires a separate verified binary label column, for example --target-column diabetes.")
        if not rows[args.target_column].isin([0, 1]).all():
            raise ValueError("Diabetes risk labels must be 0 (no diabetes) or 1 (diabetes).")
        classes = 2
    else:
        if not rows[args.target_column].isin([0, 1, 2, 3, 4]).all():
            raise ValueError("DR grading labels must be integers from 0 to 4.")
        classes = 5
    if rows[args.target_column].nunique() < 2:
        raise ValueError("Training requires at least two label classes.")
    train_rows, val_rows = train_test_split(rows, test_size=args.val_size, random_state=args.seed, stratify=rows[args.target_column])

    train_set = RetinalDataset(train_rows, data_dir / args.images, args.image_size, args.target_column, augment=True)
    val_set = RetinalDataset(val_rows, data_dir / args.images, args.image_size, args.target_column)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallRetinalCNN(classes=classes).to(device)
    counts = np.bincount(train_rows[args.target_column].to_numpy(), minlength=classes)
    weights = len(train_rows) / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_score = -1.0
    history = []

    print(f"Training {len(train_set)} images; validating {len(val_set)} images on {device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_metrics = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} val_kappa={val_metrics['quadratic_kappa']:.3f}")
        score = val_metrics["balanced_accuracy"] if args.task == "diabetes_risk" else val_metrics["quadratic_kappa"]
        if score > best_score:
            best_score = score
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "image_size": args.image_size, "classes": classes, "task": args.task, "target_column": args.target_column, "history": history}, output_path)

    print(json.dumps({"checkpoint": args.output, "task": args.task, "best_validation_score": best_score, "history": history}, indent=2))


if __name__ == "__main__":
    main()
