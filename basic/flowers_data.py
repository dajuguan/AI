"""
Utilities for downloading and loading the tf_flowers dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive


FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_ARCHIVE_NAME = "flower_photos.tgz"
FLOWERS_EXTRACTED_DIR = "flower_photos"
FLOWERS_CLASS_NAMES = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
]
FLOWERS_MEAN = (0.5, 0.5, 0.5)
FLOWERS_STD = (0.5, 0.5, 0.5)


def resolve_device(device: str) -> torch.device:
    normalized = device.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False."
        )
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported device '{device}'. Expected one of: auto, cpu, cuda."
        )
    return torch.device(normalized)


def prepare_flowers_dataset(data_dir: str) -> Path:
    root_dir = Path(data_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = root_dir / FLOWERS_EXTRACTED_DIR
    if dataset_dir.is_dir() and all(
        (dataset_dir / name).is_dir() for name in FLOWERS_CLASS_NAMES
    ):
        return dataset_dir

    download_and_extract_archive(
        url=FLOWERS_URL,
        download_root=str(root_dir),
        extract_root=str(root_dir),
        filename=FLOWERS_ARCHIVE_NAME,
        remove_finished=False,
    )
    return dataset_dir


def _split_class_indices(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
    total = len(indices)
    if total < 3:
        raise ValueError(
            "Each class needs at least 3 samples for train/val/test splits."
        )

    n_train = max(1, int(total * 0.8))
    n_val = max(1, int(total * 0.1))
    n_test = total - n_train - n_val

    while n_test < 1:
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Unable to allocate at least one test sample per class.")
        n_test = total - n_train - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    return train_indices, val_indices, test_indices


def _stratified_split_indices(
    targets: Sequence[int],
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for index, target in enumerate(targets):
        by_class.setdefault(int(target), []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in sorted(by_class):
        class_indices = by_class[class_id]
        rng.shuffle(class_indices)
        class_train, class_val, class_test = _split_class_indices(class_indices)
        train_indices.extend(class_train)
        val_indices.extend(class_val)
        test_indices.extend(class_test)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD),
        ]
    )


def _build_train_transform(
    image_size: int,
    augment_train: bool,
) -> transforms.Compose:
    if not augment_train:
        return _build_eval_transform(image_size)

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(FLOWERS_MEAN, FLOWERS_STD),
        ]
    )


def build_flowers_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    seed: int,
    augment_train: bool = False,
) -> dict[str, object]:
    dataset_dir = prepare_flowers_dataset(data_dir)
    base_dataset = datasets.ImageFolder(str(dataset_dir))
    train_indices, val_indices, test_indices = _stratified_split_indices(
        base_dataset.targets, seed=seed
    )
    print(
        f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test samples."
    )

    train_transform = _build_train_transform(
        image_size=image_size,
        augment_train=augment_train,
    )
    eval_transform = _build_eval_transform(image_size)

    train_dataset = Subset(
        datasets.ImageFolder(str(dataset_dir), transform=train_transform), train_indices
    )
    val_dataset = Subset(
        datasets.ImageFolder(str(dataset_dir), transform=eval_transform), val_indices
    )
    test_dataset = Subset(
        datasets.ImageFolder(str(dataset_dir), transform=eval_transform), test_indices
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": list(base_dataset.classes),
        "input_dim": 3 * image_size * image_size,
    }
