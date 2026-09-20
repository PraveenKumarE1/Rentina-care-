"""Train and evaluate a robust PyTorch retinal-image classifier."""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


class RetinalDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        image_dir: Path,
        image_size: int,
        target_column: str,
        augment: bool = False,
    ):
        self.rows = rows.reset_index(drop=True)
        self.image_dir = image_dir
        self.image_size = image_size
        self.target_column = target_column
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, index: int) -> Image.Image:
        row = self.rows.iloc[index]
        stem = str(row["id_code"])
        candidates = [
            self.image_dir / f"{stem}.png",
            self.image_dir / f"{stem}.jpg",
            self.image_dir / f"{stem}.jpeg",
            self.image_dir / f"{stem}.bmp",
            self.image_dir / f"{stem}.tif",
            self.image_dir / f"{stem}.tiff",
        ]
        image_path = next((path for path in candidates if path.exists()), None)
        if image_path is None:
            alternatives = list(self.image_dir.glob(f"{stem}.*"))
            if not alternatives:
                raise FileNotFoundError(f"Image not found for id_code={stem}")
            image_path = alternatives[0]
        image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        if self.augment:
            image = self._augment(image)
        image = ImageOps.fit(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.BILINEAR,
            centering=(0.5, 0.5),
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.5:
            degrees = random.uniform(-10, 10)
            image = image.rotate(degrees, resample=Image.Resampling.BILINEAR)
        if random.random() < 0.8:
            factor = random.uniform(0.75, 1.25)
            image = ImageEnhance.Color(image).enhance(factor)
        if random.random() < 0.8:
            factor = random.uniform(0.80, 1.20)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if random.random() < 0.5:
            factor = random.uniform(0.90, 1.10)
            image = ImageEnhance.Brightness(image).enhance(factor)
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self._load_image(index), int(self.rows.iloc[index][self.target_column])


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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(out_channels) if not isinstance(self.shortcut, nn.Identity) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut_bn(self.shortcut(inputs))
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        return self.relu(output + residual)


class ResidualRetinalCNN(nn.Module):
    def __init__(self, classes: int = 5, dropout: float = 0.35):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return self.classifier(self.pool(output))


def build_model(architecture: str, classes: int) -> nn.Module:
    if architecture == "small":
        return SmallRetinalCNN(classes)
    if architecture == "residual":
        return ResidualRetinalCNN(classes)
    raise ValueError(f"Unknown architecture: {architecture}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[indices]
    return mixed, labels, labels[indices], lam


def cutmix_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(images.size(0), device=images.device)
    batch_size, _, height, width = images.shape
    cut_width = int(width * np.sqrt(1.0 - lam))
    cut_height = int(height * np.sqrt(1.0 - lam))
    center_x = int(np.random.randint(width))
    center_y = int(np.random.randint(height))
    x1 = max(0, center_x - cut_width // 2)
    x2 = min(width, x1 + cut_width)
    y1 = max(0, center_y - cut_height // 2)
    y2 = min(height, y1 + cut_height)
    images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (height * width))
    return images, labels, labels[indices], lam


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
) -> Dict[str, float]:
    model.train(training)
    total_loss = 0.0
    actual: List[int] = []
    predicted: List[int] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training and mixup_alpha > 0 and cutmix_alpha > 0 and random.random() < 0.5:
            images, labels_a, labels_b, lam = cutmix_batch(images, labels, cutmix_alpha)
        elif training and mixup_alpha > 0:
            images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0
        optimizer.zero_grad(set_to_none=True) if training else None
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = lam * loss_fn(logits, labels_a) + (1.0 - lam) * loss_fn(logits, labels_b)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        predicted.extend(logits.argmax(1).detach().cpu().tolist())
        actual.extend(labels.detach().cpu().tolist())
    metrics = {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": accuracy_score(actual, predicted),
        "balanced_accuracy": balanced_accuracy_score(actual, predicted),
        "quadratic_kappa": cohen_kappa_score(actual, predicted, weights="quadratic"),
    }
    return metrics


def find_temperature(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in loader:
            logits_list.append(model(images.to(device)).cpu())
            labels_list.append(labels)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / temperature.clamp(0.05, 10.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.item())


def tta_predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    model.eval()
    tensors = [tensor]
    tensors.append(torch.flip(tensor, dims=[3]))
    tensors.append(torch.rot90(tensor, 1, dims=[2, 3]))
    probabilities = []
    with torch.no_grad():
        for item in tensors:
            logits = model(item.to(device)) / max(float(temperature), 0.05)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.mean(probabilities, axis=0)[0]


def safe_stratify(labels: np.ndarray, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.Series(labels).value_counts()
    can_stratify = bool((counts >= 2).all()) and len(counts) >= 2
    return train_test_split(
        pd.DataFrame({"index": np.arange(len(labels))}),
        test_size=test_size,
        random_state=seed,
        stratify=labels if can_stratify else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a robust retinal-image classifier on verified labels")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--csv", default="train.csv")
    parser.add_argument("--images", default="train_images")
    parser.add_argument("--output", default="models/user_retinal_cnn.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-column", default="diagnosis")
    parser.add_argument("--task", choices=["dr_grading", "diabetes_risk"], default="dr_grading")
    parser.add_argument("--architecture", choices=["small", "residual"], default="residual")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    rows = pd.read_csv(data_dir / args.csv)
    required = {"id_code", args.target_column}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    rows[args.target_column] = pd.to_numeric(rows[args.target_column], errors="raise").astype(int)
    if args.task == "diabetes_risk":
        if args.target_column == "diagnosis":
            raise ValueError("Diabetes risk requires a verified binary label column such as diabetes.")
        if not rows[args.target_column].isin([0, 1]).all():
            raise ValueError("Diabetes risk labels must be 0 or 1.")
        classes = 2
    else:
        if not rows[args.target_column].isin([0, 1, 2, 3, 4]).all():
            raise ValueError("DR grading labels must be integers from 0 to 4.")
        classes = 5
    if rows[args.target_column].nunique() < 2:
        raise ValueError("Training requires at least two label classes.")
    train_rows, val_rows = safe_stratify(rows[args.target_column].to_numpy(), args.val_size, args.seed)
    train_rows = rows.iloc[train_rows["index"].to_numpy()].reset_index(drop=True)
    val_rows = rows.iloc[val_rows["index"].to_numpy()].reset_index(drop=True)

    train_set = RetinalDataset(train_rows, data_dir / args.images, args.image_size, args.target_column, augment=True)
    val_set = RetinalDataset(val_rows, data_dir / args.images, args.image_size, args.target_column, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(args.architecture, classes).to(device)
    counts = np.bincount(train_rows[args.target_column].to_numpy(), minlength=classes)
    weights = len(train_rows) / np.maximum(counts, 1)
    weights = torch.tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_score = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, object]] = []
    epochs_without_improvement = 0

    print(f"Training {len(train_set)} images; validating {len(val_set)} images on {device}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer, device, True, args.mixup, args.cutmix)
        val_metrics = run_epoch(model, val_loader, loss_fn, None, device, False)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(
            f"epoch={epoch:02d} train_loss={train_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_kappa={val_metrics['quadratic_kappa']:.3f}"
        )
        score = val_metrics["balanced_accuracy"] if args.task == "diabetes_risk" else val_metrics["quadratic_kappa"]
        if score > best_score + 1e-6:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    model.load_state_dict(best_state or model.state_dict())
    temperature = find_temperature(model, val_loader, device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "architecture": args.architecture,
        "image_size": args.image_size,
        "classes": classes,
        "task": args.task,
        "target_column": args.target_column,
        "temperature": temperature,
        "normalization": "centered",
        "enhancement": "none",
        "history": history,
        "best_validation_score": best_score,
        "class_counts": counts.astype(int).tolist(),
    }
    torch.save(checkpoint, output_path)
    print(json.dumps({"checkpoint": str(output_path), "task": args.task, "best_validation_score": best_score, "temperature": temperature, "history": history}, indent=2))


if __name__ == "__main__":
    main()
