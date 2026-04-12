"""
Train a simple CNN on tf_flowers and inspect convolutional feature maps.

Example:
    python3 basic/flowers_cnn.py --epochs 15 --batch-size 64 --device auto
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

try:
    from flowers_data import build_flowers_dataloaders, resolve_device
except ImportError:  # pragma: no cover - allows module import from the repo root.
    from basic.flowers_data import build_flowers_dataloaders, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts/flowers_cnn")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["none", "cosine"],
        default="cosine",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--device", type=str, choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k-feature-maps", type=int, default=6)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (
        ModuleNotFoundError
    ) as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError(
            "matplotlib is required for plotting outputs. Install it with "
            "`python3 -m pip install matplotlib` and rerun the script."
        ) from exc
    return plt


def load_existing_metrics(metrics_path: Path) -> dict[str, object] | None:
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


class FlowerCNN(nn.Module):
    class ConvStage(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)

    def __init__(self, image_size: int, num_classes: int) -> None:
        super().__init__()
        del image_size  # The classifier head uses adaptive pooling, so image_size is unused.
        self.stage1 = self.ConvStage(3, 32)
        self.stage2 = self.ConvStage(32, 64)
        self.stage3 = self.ConvStage(64, 128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.activation = nn.ReLU(inplace=True)

    def _forward_convs(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conv1 = self.stage1(x)
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        return conv1, conv2, conv3

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        conv1, conv2, conv3 = self._forward_convs(x)
        flattened = torch.flatten(self.global_pool(conv3), start_dim=1)
        hidden = self.activation(self.fc1(flattened))
        logits = self.fc2(hidden)
        if return_features:
            return logits, {
                "conv1": conv1,
                "conv2": conv2,
                "conv3": conv3,
                "hidden": hidden,
            }
        return logits


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    collect_outputs: bool = False,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    saved_images: list[torch.Tensor] = []
    saved_logits: list[torch.Tensor] = []
    saved_labels: list[torch.Tensor] = []
    saved_preds: list[torch.Tensor] = []
    saved_hidden: list[torch.Tensor] = []
    saved_conv1: list[torch.Tensor] = []
    saved_conv2: list[torch.Tensor] = []
    saved_conv3: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits, features = model(images, return_features=True)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if collect_outputs:
                saved_images.append(images.cpu())
                saved_logits.append(logits.cpu())
                saved_labels.append(labels.cpu())
                saved_preds.append(preds.cpu())
                saved_hidden.append(features["hidden"].cpu())
                saved_conv1.append(features["conv1"].cpu())
                saved_conv2.append(features["conv2"].cpu())
                saved_conv3.append(features["conv3"].cpu())

    metrics = {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }
    if not collect_outputs:
        return metrics, None

    outputs = {
        "images": torch.cat(saved_images, dim=0),
        "logits": torch.cat(saved_logits, dim=0),
        "labels": torch.cat(saved_labels, dim=0),
        "preds": torch.cat(saved_preds, dim=0),
        "hidden": torch.cat(saved_hidden, dim=0),
        "conv1": torch.cat(saved_conv1, dim=0),
        "conv2": torch.cat(saved_conv2, dim=0),
        "conv3": torch.cat(saved_conv3, dim=0),
    }
    return metrics, outputs


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    return (image * 0.5 + 0.5).clamp(0.0, 1.0)


def select_random_class_examples(
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    seed: int,
) -> tuple[list[int], list[dict[str, object]]]:
    rng = random.Random(seed)
    available_classes = sorted(set(int(label) for label in labels.tolist()))
    rng.shuffle(available_classes)
    selected_classes = available_classes[: min(5, len(available_classes))]

    selected_indices: list[int] = []
    selection_info: list[dict[str, object]] = []
    for class_index in selected_classes:
        candidates = (
            torch.nonzero(labels == class_index, as_tuple=False).flatten().tolist()
        )
        selected_index = rng.choice(candidates)
        selected_indices.append(selected_index)
        selection_info.append(
            {
                "sample_index": selected_index,
                "true_class": class_names[class_index],
                "predicted_class": class_names[int(preds[selected_index])],
            }
        )
    return selected_indices, selection_info


def normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    feature_map = feature_map - feature_map.min()
    feature_map = feature_map / feature_map.max().clamp_min(1e-6)
    return feature_map


def plot_selected_examples(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    selected_indices: list[int],
    output_path: Path,
    plt,
) -> None:
    fig, axes = plt.subplots(
        1, len(selected_indices), figsize=(3.0 * len(selected_indices), 3.8)
    )
    if len(selected_indices) == 1:
        axes = [axes]

    for axis, sample_index in zip(axes, selected_indices):
        image = unnormalize_image(images[sample_index]).permute(1, 2, 0).numpy()
        true_name = class_names[int(labels[sample_index])]
        pred_name = class_names[int(preds[sample_index])]

        axis.imshow(image)
        if true_name == pred_name:
            axis.set_title(true_name, fontsize=11)
        else:
            axis.set_title(f"{true_name}\n-> {pred_name}", fontsize=11, color="crimson")
        axis.axis("off")

    fig.suptitle("Random 5-Class CNN Predictions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_layer_feature_maps(
    layer_name: str,
    feature_maps: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    selected_indices: list[int],
    output_path: Path,
    top_k: int,
    plt,
) -> None:
    n_rows = len(selected_indices)
    n_cols = top_k + 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.3 * n_rows),
        squeeze=False,
    )

    for row, sample_index in enumerate(selected_indices):
        original = unnormalize_image(images[sample_index]).permute(1, 2, 0).numpy()
        true_name = class_names[int(labels[sample_index])]
        pred_name = class_names[int(preds[sample_index])]

        axes[row, 0].imshow(original)
        if true_name == pred_name:
            axes[row, 0].set_title(true_name, fontsize=10)
        else:
            axes[row, 0].set_title(
                f"{true_name}\n-> {pred_name}", fontsize=10, color="crimson"
            )
        axes[row, 0].axis("off")

        sample_features = feature_maps[sample_index]
        channel_scores = sample_features.mean(dim=(1, 2))
        num_channels = min(top_k, sample_features.shape[0])
        top_channels = torch.topk(channel_scores, k=num_channels).indices.tolist()

        for col in range(1, n_cols):
            axes[row, col].axis("off")
            if col - 1 >= num_channels:
                continue
            channel_index = top_channels[col - 1]
            feature_map = normalize_feature_map(sample_features[channel_index])
            axes[row, col].imshow(feature_map.numpy(), cmap="viridis")
            axes[row, col].set_title(f"ch {channel_index}", fontsize=8)
            axes[row, col].axis("off")

    fig.suptitle(f"{layer_name} Feature Maps", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_train_history(history: list[dict[str, float]], output_path: Path, plt) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = require_pyplot()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    data = build_flowers_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed,
        augment_train=True,
    )
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    test_loader = data["test_loader"]
    class_names = data["class_names"]

    model = FlowerCNN(
        image_size=args.image_size,
        num_classes=len(class_names),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.epochs > 0 and args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )
    best_model_path = output_dir / "best_model.pt"
    metrics_path = output_dir / "metrics.json"

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    existing_metrics = load_existing_metrics(metrics_path)
    if existing_metrics is not None and args.epochs == 0:
        history = list(existing_metrics.get("history", []))
        best_val_accuracy = float(existing_metrics.get("best_val_accuracy", -1.0))
        best_epoch = int(existing_metrics.get("best_epoch", -1))

    if args.epochs > 0:
        for epoch in range(1, args.epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_metrics, _ = evaluate(model, val_loader, criterion, device)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "lr": current_lr,
                }
            )
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                f"lr={current_lr:.6f}"
            )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)

            if scheduler is not None:
                scheduler.step()
    elif best_model_path.exists():
        print(f"Skipping training and reusing checkpoint: {best_model_path}")
    else:
        raise RuntimeError(
            "No checkpoint found for evaluation-only mode. Run with --epochs > 0 first."
        )

    if not best_model_path.exists():
        raise RuntimeError(f"Expected checkpoint to exist at {best_model_path}.")

    state_dict = torch.load(best_model_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with the current CNN architecture. "
            "Run with --epochs > 0 to train a new CNN checkpoint."
        ) from exc

    if best_val_accuracy < 0.0:
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        best_val_accuracy = val_metrics["accuracy"]
        if best_epoch < 0:
            best_epoch = 0

    test_metrics, outputs = evaluate(
        model,
        test_loader,
        criterion,
        device,
        collect_outputs=True,
    )
    if outputs is None:
        raise RuntimeError("Expected test outputs to be collected.")

    selected_indices, selection_info = select_random_class_examples(
        labels=outputs["labels"],
        preds=outputs["preds"],
        class_names=class_names,
        seed=args.seed,
    )

    if history:
        plot_train_history(history, output_dir / "train_history.png", plt)
    plot_selected_examples(
        images=outputs["images"],
        labels=outputs["labels"],
        preds=outputs["preds"],
        class_names=class_names,
        selected_indices=selected_indices,
        output_path=output_dir / "random_class_predictions.png",
        plt=plt,
    )
    for layer_name in ["conv1", "conv2", "conv3"]:
        plot_layer_feature_maps(
            layer_name=layer_name,
            feature_maps=outputs[layer_name],
            images=outputs["images"],
            labels=outputs["labels"],
            preds=outputs["preds"],
            class_names=class_names,
            selected_indices=selected_indices,
            output_path=output_dir / f"{layer_name}_feature_maps.png",
            top_k=args.top_k_feature_maps,
            plt=plt,
        )

    torch.save(
        {
            "class_names": class_names,
            "selection_info": selection_info,
            "images": outputs["images"][selected_indices],
            "labels": outputs["labels"][selected_indices],
            "preds": outputs["preds"][selected_indices],
            "hidden": outputs["hidden"][selected_indices],
            "conv1": outputs["conv1"][selected_indices],
            "conv2": outputs["conv2"][selected_indices],
            "conv3": outputs["conv3"][selected_indices],
        },
        output_dir / "selected_features.pt",
    )

    metrics = {
        "config": {
            "data_dir": args.data_dir,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "image_size": args.image_size,
            "device": str(device),
            "seed": args.seed,
            "top_k_feature_maps": args.top_k_feature_maps,
            "augment_train": True,
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "random_class_examples": selection_info,
        "history": history,
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(
        f"Finished. best_val_acc={best_val_accuracy:.4f}, "
        f"test_acc={test_metrics['accuracy']:.4f}. Outputs saved to {output_dir}"
    )


if __name__ == "__main__":
    main()
