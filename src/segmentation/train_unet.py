"""U-Net training utilities for Phase 2."""

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data_prep.load_herlev import list_herlev_images, load_image_and_masks
from src.segmentation.unet_model import UNet
from src.utils.config import DATA_RAW_HERLEV, RANDOM_SEED, RESULTS_TABLES, ROOT_DIR


INPUT_SIZE = (128, 128)
N_CLASSES = 3


def _resize_with_padding(
    image: Image.Image,
    size: Tuple[int, int] = INPUT_SIZE,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
    fill: int | Tuple[int, int, int] = 0,
) -> Image.Image:
    """
    Resize isotropically and center-pad to ``size``.

    This preserves aspect ratio. We intentionally avoid direct stretch/squash
    resize because anisotropic scaling would distort downstream shape metrics
    such as nucleus circularity.
    """
    target_width, target_height = size
    scale = min(target_width / image.width, target_height / image.height)
    resized_width = max(1, int(round(image.width * scale)))
    resized_height = max(1, int(round(image.height * scale)))
    resized = image.resize((resized_width, resized_height), resample)
    canvas = Image.new(image.mode, size, fill)
    offset = ((target_width - resized_width) // 2, (target_height - resized_height) // 2)
    canvas.paste(resized, offset)
    return canvas


def _resize_rgb_image(image: np.ndarray, size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    image_pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    resized = _resize_with_padding(image_pil, size=size, resample=Image.Resampling.BILINEAR, fill=(0, 0, 0))
    return np.asarray(resized, dtype=np.float32)


def _resize_bool_mask(mask: np.ndarray, size: Tuple[int, int] = INPUT_SIZE) -> np.ndarray:
    mask_pil = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    resized = _resize_with_padding(mask_pil, size=size, resample=Image.Resampling.NEAREST, fill=0)
    return np.asarray(resized, dtype=np.uint8) > 0


def preprocess_rgb_image(image: np.ndarray, size: Tuple[int, int] = INPUT_SIZE) -> torch.Tensor:
    """
    Convert a Herlev RGB image to a model input tensor.

    Locked preprocessing:
        - keep RGB; do not convert to grayscale
        - resize isotropically with aspect-ratio preservation, then center-pad
          to ``size`` using black pixels
        - normalize only from uint8 ``[0, 255]`` to float ``[0, 1]``
        - return ``3 x H x W``
    """
    resized = _resize_rgb_image(image, size=size) / 255.0
    return torch.from_numpy(np.transpose(resized, (2, 0, 1))).float()


def build_segmentation_target(
    nucleus_mask: np.ndarray,
    cytoplasm_mask: np.ndarray,
    size: Tuple[int, int] = INPUT_SIZE,
) -> torch.Tensor:
    """
    Build a three-class target mask.

    Class mapping:
        0 = background/other
        1 = cytoplasm
        2 = nucleus

    Merge rule:
        - pixels where ``nucleus_mask == 1`` become class 2
        - pixels where ``cytoplasm_mask == 1`` and ``nucleus_mask == 0`` become
          class 1
        - all remaining pixels become class 0

    Nucleus and cytoplasm masks are expected to be pixel-exclusive. Masks are
    resized isotropically and center-padded with nearest-neighbor interpolation
    so class labels and aspect ratio are preserved.
    """
    nucleus = _resize_bool_mask(nucleus_mask, size=size)
    cytoplasm = _resize_bool_mask(cytoplasm_mask, size=size)
    if np.logical_and(nucleus, cytoplasm).any():
        raise ValueError("nucleus_mask and cytoplasm_mask must be pixel-exclusive after resizing.")

    target = np.zeros(size[::-1], dtype=np.int64)
    target[np.logical_and(cytoplasm, ~nucleus)] = 1
    target[nucleus] = 2
    return torch.from_numpy(target).long()


class HerlevSegmentationDataset(Dataset):
    """Dataset that returns RGB image tensors and three-class segmentation targets."""

    def __init__(
        self,
        raw_dir: Path = DATA_RAW_HERLEV,
        image_ids: Optional[Sequence[str]] = None,
        image_size: Tuple[int, int] = INPUT_SIZE,
    ):
        self.raw_dir = Path(raw_dir)
        self.image_size = image_size
        images = list_herlev_images(self.raw_dir)
        if image_ids is not None:
            allowed = set(image_ids)
            images = [path for path in images if path.stem in allowed or path.name in allowed]
        self.image_paths: List[Path] = images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        image, nucleus_mask, cytoplasm_mask = load_image_and_masks(image_path)
        image_tensor = preprocess_rgb_image(image, size=self.image_size)
        target_tensor = build_segmentation_target(nucleus_mask, cytoplasm_mask, size=self.image_size)
        return image_tensor, target_tensor


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, n_classes: int = N_CLASSES) -> torch.Tensor:
    """Soft Dice loss averaged over foreground classes: cytoplasm and nucleus."""
    probs = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    smooth = 1e-6
    intersection = torch.sum(probs * one_hot, dims)
    cardinality = torch.sum(probs + one_hot, dims)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice[1:].mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CrossEntropy + foreground Dice loss."""
    return F.cross_entropy(logits, targets) + dice_loss(logits, targets)


def _batch_foreground_dice_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int = N_CLASSES,
) -> Tuple[float, float]:
    dice_scores = []
    iou_scores = []
    for cls in range(1, n_classes):
        pred = predictions == cls
        gt = targets == cls
        intersection = torch.logical_and(pred, gt).sum().float()
        pred_sum = pred.sum().float()
        gt_sum = gt.sum().float()
        union = torch.logical_or(pred, gt).sum().float()
        dice = (2.0 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice_scores.append(float(dice.item()))
        iou_scores.append(float(iou.item()))
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))


def _resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_unet(
    train_ids: Optional[Iterable[str]],
    val_ids: Optional[Iterable[str]],
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 8,
    raw_dir: Path = DATA_RAW_HERLEV,
    image_size: Tuple[int, int] = INPUT_SIZE,
    device: Optional[str] = None,
    base_channels: int = 32,
    early_stopping_patience: Optional[int] = 10,
    min_delta: float = 1e-4,
    checkpoint_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> List[dict]:
    """
    Train the RGB U-Net and log validation Dice/IoU after every epoch.

    Loss is CrossEntropy + foreground Dice loss. Validation Dice and IoU are
    mean scores over foreground classes only: cytoplasm and nucleus.

    Validation gate: do not proceed past Phase 2 until validation Dice > 0.85.
    """
    _seed_everything(RANDOM_SEED)
    device_obj = _resolve_device(device)

    train_dataset = HerlevSegmentationDataset(raw_dir=raw_dir, image_ids=list(train_ids) if train_ids else None, image_size=image_size)
    val_dataset = HerlevSegmentationDataset(raw_dir=raw_dir, image_ids=list(val_ids) if val_ids else None, image_size=image_size)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Empty train/validation dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=3, n_classes=N_CLASSES, base_channels=base_channels).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_dice = -1.0
    epochs_without_improvement = 0
    history: List[dict] = []

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else ROOT_DIR / "results" / "unet_best.pt"
    log_path = Path(log_path) if log_path else RESULTS_TABLES / "unet_training_log.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        n_train_batches = 0
        for images, targets in train_loader:
            images = images.to(device_obj)
            targets = targets.to(device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            n_train_batches += 1

        model.eval()
        val_loss_total = 0.0
        val_dice_scores = []
        val_iou_scores = []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device_obj)
                targets = targets.to(device_obj)
                logits = model(images)
                loss = segmentation_loss(logits, targets)
                predictions = torch.argmax(logits, dim=1)
                dice, iou = _batch_foreground_dice_iou(predictions, targets)
                val_loss_total += float(loss.item())
                val_dice_scores.append(dice)
                val_iou_scores.append(iou)

        row = {
            "epoch": epoch,
            "train_loss": train_loss_total / max(1, n_train_batches),
            "val_loss": val_loss_total / max(1, len(val_loader)),
            "val_dice": float(np.mean(val_dice_scores)),
            "val_iou": float(np.mean(val_iou_scores)),
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "batch_size": batch_size,
            "image_size": list(image_size),
            "device": str(device_obj),
            "base_channels": base_channels,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{epochs:03d} "
            f"train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_dice={row['val_dice']:.4f} val_iou={row['val_iou']:.4f}"
        )

        improved = row["val_dice"] > best_val_dice + min_delta
        if improved:
            best_val_dice = row["val_dice"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_val_dice,
                    "config": {
                        "in_channels": 3,
                        "n_classes": N_CLASSES,
                        "base_channels": base_channels,
                        "image_size": list(image_size),
                    },
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"early stopping at epoch {epoch:03d}: "
                f"best_val_dice={best_val_dice:.4f}, patience={early_stopping_patience}"
            )
            break

    return history


if __name__ == "__main__":
    # TODO: load train/val ids from the group split output, then call train_unet.
    pass
