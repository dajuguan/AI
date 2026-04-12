"""
Train a simple MLP on tf_flowers and inspect hidden-layer representations.

Example:
    python3 basic/flowers_mlp.py --epochs 15 --batch-size 64 --device auto
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
    parser.add_argument("--output-dir", type=str, default="./artifacts/flowers_mlp")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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


class FlowerMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.flatten(x)
        hidden1 = self.activation(self.fc1(x))
        hidden2 = self.activation(self.fc2(hidden1))
        logits = self.fc3(hidden2)
        if return_features:
            return logits, {"hidden1": hidden1, "hidden2": hidden2}
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

    saved_logits: list[torch.Tensor] = []
    saved_labels: list[torch.Tensor] = []
    saved_preds: list[torch.Tensor] = []
    saved_images: list[torch.Tensor] = []
    saved_hidden1: list[torch.Tensor] = []
    saved_hidden2: list[torch.Tensor] = []

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
                saved_hidden1.append(features["hidden1"].cpu())
                saved_hidden2.append(features["hidden2"].cpu())

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
        "hidden1": torch.cat(saved_hidden1, dim=0),
        "hidden2": torch.cat(saved_hidden2, dim=0),
    }
    return metrics, outputs


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


def project_hidden2_templates_to_input(
    model: FlowerMLP,
    hidden2: torch.Tensor,
    preds: torch.Tensor,
) -> torch.Tensor:
    fc3_rows = model.fc3.weight.detach().cpu()[preds]
    fc2_weight = model.fc2.weight.detach().cpu()
    fc1_weight = model.fc1.weight.detach().cpu()

    hidden2_evidence = torch.relu(hidden2 * fc3_rows)
    hidden1_projection = hidden2_evidence @ fc2_weight
    input_projection = hidden1_projection @ fc1_weight
    return input_projection


def project_hidden1_templates_to_input(
    model: FlowerMLP,
    hidden1: torch.Tensor,
    preds: torch.Tensor,
) -> torch.Tensor:
    fc3_rows = model.fc3.weight.detach().cpu()[preds]
    fc2_weight = model.fc2.weight.detach().cpu()
    fc1_weight = model.fc1.weight.detach().cpu()

    hidden1_class_weights = fc3_rows @ fc2_weight
    hidden1_evidence = torch.relu(hidden1 * hidden1_class_weights)
    input_projection = hidden1_evidence @ fc1_weight
    return input_projection


def normalize_template_image(
    template_flat: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    image = template_flat.view(3, image_size, image_size)
    image = image - image.min()
    image = image / image.max().clamp_min(1e-6)
    return image


def denormalize_input_image(image: torch.Tensor) -> torch.Tensor:
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


def plot_random_class_templates(
    model: FlowerMLP,
    hidden1: torch.Tensor,
    hidden2: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    image_size: int,
    selected_indices: list[int],
    output_path: Path,
    plt,
) -> None:
    hidden1_template_inputs = project_hidden1_templates_to_input(
        model=model,
        hidden1=hidden1[selected_indices],
        preds=preds[selected_indices],
    )
    hidden2_template_inputs = project_hidden2_templates_to_input(
        model=model,
        hidden2=hidden2[selected_indices],
        preds=preds[selected_indices],
    )

    fig, axes = plt.subplots(
        3,
        len(selected_indices) + 1,
        figsize=(3.0 * len(selected_indices) + 2.0, 10.2),
        gridspec_kw={"width_ratios": [0.9] + [3.0] * len(selected_indices)},
        squeeze=False,
    )

    for row, row_label in enumerate(["hidden1", "hidden2", "real"]):
        axes[row, 0].axis("off")
        axes[row, 0].text(
            0.5,
            0.5,
            row_label,
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#f1f3f5",
                "edgecolor": "#c7ced6",
            },
        )

    for col, sample_index in enumerate(selected_indices):
        true_name = class_names[int(labels[sample_index])]
        pred_name = class_names[int(preds[sample_index])]
        title = true_name if true_name == pred_name else f"{true_name}\n-> {pred_name}"
        title_color = "black" if true_name == pred_name else "crimson"

        hidden1_image = normalize_template_image(
            hidden1_template_inputs[col], image_size=image_size
        )
        hidden2_image = normalize_template_image(
            hidden2_template_inputs[col], image_size=image_size
        )

        axes[0, col + 1].imshow(hidden1_image.permute(1, 2, 0).numpy())
        axes[0, col + 1].set_title(title, fontsize=11, color=title_color)
        axes[0, col + 1].axis("off")

        axes[1, col + 1].imshow(hidden2_image.permute(1, 2, 0).numpy())
        axes[1, col + 1].axis("off")

        real_image = denormalize_input_image(images[sample_index])
        axes[2, col + 1].imshow(real_image.permute(1, 2, 0).numpy())
        axes[2, col + 1].axis("off")

    fig.suptitle("Hidden Templates", fontsize=16)
    fig.subplots_adjust(
        left=0.03,
        right=0.995,
        top=0.90,
        bottom=0.03,
        wspace=0.06,
        hspace=0.06,
    )
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
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
    input_dim = data["input_dim"]

    model = FlowerMLP(input_dim=input_dim, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    model.load_state_dict(state_dict)

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

    if history:
        plot_train_history(history, output_dir / "train_history.png", plt)
    selected_indices, template_selection = select_random_class_examples(
        labels=outputs["labels"],
        preds=outputs["preds"],
        class_names=class_names,
        seed=args.seed,
    )
    plot_random_class_templates(
        model=model,
        hidden1=outputs["hidden1"],
        hidden2=outputs["hidden2"],
        images=outputs["images"],
        labels=outputs["labels"],
        preds=outputs["preds"],
        class_names=class_names,
        image_size=args.image_size,
        selected_indices=selected_indices,
        output_path=output_dir / "random_class_templates.png",
        plt=plt,
    )
    torch.save(
        {
            "class_names": class_names,
            "labels": outputs["labels"],
            "preds": outputs["preds"],
            "logits": outputs["logits"],
            "hidden2": outputs["hidden2"],
        },
        output_dir / "activations.pt",
    )

    metrics = {
        "config": {
            "data_dir": args.data_dir,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "image_size": args.image_size,
            "device": str(device),
            "seed": args.seed,
            "augment_train": True,
        },
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "random_class_templates": template_selection,
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
