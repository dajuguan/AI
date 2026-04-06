"""
A simple sin(x) function to demonstrate that NN can actually learn the underlying structure instead of memorizing, to extrapolate beyond the training range, if the right architecture is used.
"""

import math
import random

import torch


N_TRAIN_IN_RANGE = 2000
N_TRAIN_OUT_OF_RANGE = 20
TRAIN_INNER_RADIUS = math.pi
TRAIN_OUTSIDE_MULTIPLIER = 10.0
EVAL_MULTIPLIER = 20.0
N_EVAL_SAMPLES = 10000
TRAIN_STEPS = 3000
LEARNING_RATE = 3e-3
LR_WARMUP_FRACTION = 0.05
MIN_LR_SCALE = 0.05
WEIGHT_DECAY = 1e-5
STATE_DIM = 2
SEED = 0
DEVICE = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' was requested, but CUDA is unavailable.")
    return device


class InputScaling(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        if scale <= 0.0:
            raise ValueError("Input scaling must be positive.")
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


class MatrixExpEmbedding(torch.nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        input_scale: float = TRAIN_INNER_RADIUS,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("STATE_DIM must be positive.")
        self.state_dim = state_dim
        self.input_scaler = InputScaling(input_scale)
        self.dynamics = torch.nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.initial_state = torch.nn.Parameter(torch.randn(state_dim, 1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_x = self.input_scaler(x).reshape(-1)
        transition = torch.matrix_exp(
            scaled_x[:, None, None] * self.dynamics[None, :, :]
        )
        state = transition @ self.initial_state
        return state.squeeze(-1)

    def learned_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "dynamics": self.dynamics.detach().cpu(),
            "initial_state": self.initial_state.detach().cpu().squeeze(1),
        }


class EmbeddedReLUMLP(torch.nn.Module):
    def __init__(
        self,
        state_dim: int = STATE_DIM,
    ) -> None:
        super().__init__()
        self.embedding = MatrixExpEmbedding(
            state_dim=state_dim,
            input_scale=TRAIN_INNER_RADIUS,
        )
        self.output_head = torch.nn.Linear(state_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.embedding(x))

    def learned_parameters(self) -> dict[str, torch.Tensor]:
        params = self.embedding.learned_parameters()
        params["output_weight"] = self.output_head.weight.detach().cpu().squeeze(0)
        params["output_bias"] = self.output_head.bias.detach().cpu()
        return params


def sample_outside_points(
    count: int,
    inner_radius: float,
    outer_radius: float,
    device: torch.device,
) -> torch.Tensor:
    if count < 0:
        raise ValueError("N_TRAIN_OUT_OF_RANGE must be non-negative.")
    if outer_radius <= inner_radius:
        raise ValueError("The outer radius must be larger than the inner radius.")
    if count == 0:
        return torch.empty((0, 1), device=device)

    span = outer_radius - inner_radius
    left_count = count // 2
    right_count = count - left_count
    chunks: list[torch.Tensor] = []

    if left_count > 0:
        left = -inner_radius - torch.rand((left_count, 1), device=device) * span
        chunks.append(left)
    if right_count > 0:
        right = inner_radius + torch.rand((right_count, 1), device=device) * span
        chunks.append(right)

    return torch.cat(chunks, dim=0)


def make_train_data(
    n_train_in_range: int = N_TRAIN_IN_RANGE,
    n_train_out_of_range: int = N_TRAIN_OUT_OF_RANGE,
    inner_radius: float = TRAIN_INNER_RADIUS,
    outside_multiplier: float = TRAIN_OUTSIDE_MULTIPLIER,
    device: str | torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    device_obj = torch.device(device)
    train_outer_radius = inner_radius * outside_multiplier
    if outside_multiplier <= 1.0:
        raise ValueError("TRAIN_OUTSIDE_MULTIPLIER must be greater than 1.")
    if n_train_in_range < 0:
        raise ValueError("N_TRAIN_IN_RANGE must be non-negative.")

    if n_train_in_range == 0:
        x_inside = torch.empty((0, 1), device=device_obj)
    else:
        x_inside = torch.linspace(
            -inner_radius,
            inner_radius,
            n_train_in_range,
            device=device_obj,
        ).unsqueeze(1)

    x_outside = sample_outside_points(
        n_train_out_of_range,
        inner_radius,
        train_outer_radius,
        device_obj,
    )

    x_train = torch.cat([x_inside, x_outside], dim=0)
    if x_train.numel() == 0:
        raise ValueError("Training data is empty.")

    permutation = torch.randperm(x_train.shape[0], device=device_obj)
    x_train = x_train[permutation]
    y_train = torch.sin(x_train)
    return x_train, y_train, train_outer_radius


def make_eval_data(
    n_eval_samples: int = N_EVAL_SAMPLES,
    inner_radius: float = TRAIN_INNER_RADIUS,
    eval_multiplier: float = EVAL_MULTIPLIER,
    device: str | torch.device = DEVICE,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if n_eval_samples <= 0:
        raise ValueError("N_EVAL_SAMPLES must be positive.")
    if eval_multiplier <= 1.0:
        raise ValueError("EVAL_MULTIPLIER must be greater than 1.")

    device_obj = torch.device(device)
    eval_outer_radius = inner_radius * eval_multiplier
    x_eval = (
        torch.rand((n_eval_samples, 1), device=device_obj) * 2.0 - 1.0
    ) * eval_outer_radius
    y_eval = torch.sin(x_eval)
    return x_eval, y_eval, eval_outer_radius


def train_model(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    train_steps: int = TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
) -> float:
    if train_steps <= 0:
        raise ValueError("TRAIN_STEPS must be positive.")
    if not 0.0 <= LR_WARMUP_FRACTION < 1.0:
        raise ValueError("LR_WARMUP_FRACTION must be in [0, 1).")
    if not 0.0 < MIN_LR_SCALE <= 1.0:
        raise ValueError("MIN_LR_SCALE must be in (0, 1].")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY,
    )
    warmup_steps = int(train_steps * LR_WARMUP_FRACTION)

    def get_lr_for_step(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            warmup_progress = (step + 1) / warmup_steps
            return learning_rate * warmup_progress

        decay_steps = max(1, train_steps - warmup_steps)
        decay_progress = (step - warmup_steps) / decay_steps
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        min_lr = learning_rate * MIN_LR_SCALE
        return min_lr + (learning_rate - min_lr) * cosine_scale

    model.train()
    for step in range(train_steps):
        current_lr = get_lr_for_step(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        prediction = model(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_prediction = model(x_train)
        final_train_mse = torch.mean((final_prediction - y_train) ** 2).item()
    return final_train_mse


def summarize_region(
    squared_error: torch.Tensor,
    absolute_error: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    count = int(mask.sum().item())
    if count == 0:
        return {"count": 0, "mse": float("nan"), "mae": float("nan")}

    return {
        "count": count,
        "mse": squared_error[mask].mean().item(),
        "mae": absolute_error[mask].mean().item(),
    }


def evaluate_model(
    model: torch.nn.Module,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    train_outer_radius: float,
    inner_radius: float = TRAIN_INNER_RADIUS,
) -> dict[str, object]:
    model.eval()
    with torch.no_grad():
        prediction = model(x_eval).squeeze(1)

    x_values = x_eval.squeeze(1)
    targets = y_eval.squeeze(1)
    errors = prediction - targets
    squared_error = errors.square()
    absolute_error = errors.abs()

    inside_mask = x_values.abs() <= inner_radius
    seen_outer_mask = (x_values.abs() > inner_radius) & (
        x_values.abs() <= train_outer_radius
    )
    far_outer_mask = x_values.abs() > train_outer_radius

    return {
        "overall": {
            "count": int(x_values.numel()),
            "mse": squared_error.mean().item(),
            "mae": absolute_error.mean().item(),
        },
        "regions": {
            "inside": summarize_region(squared_error, absolute_error, inside_mask),
            "seen_outer": summarize_region(
                squared_error, absolute_error, seen_outer_mask
            ),
            "far_outer": summarize_region(
                squared_error, absolute_error, far_outer_mask
            ),
        },
    }


def evaluate_probe_points(
    model: torch.nn.Module,
    points: torch.Tensor,
    device: torch.device,
) -> list[dict[str, float]]:
    model.eval()
    x = points.to(device).unsqueeze(1)
    with torch.no_grad():
        prediction = model(x).squeeze(1).cpu()

    targets = torch.sin(points.cpu())
    absolute_error = (prediction - targets).abs()
    return [
        {
            "x": float(x_value),
            "target": float(target),
            "prediction": float(pred),
            "abs_error": float(abs_err),
        }
        for x_value, target, pred, abs_err in zip(
            points.tolist(),
            targets.tolist(),
            prediction.tolist(),
            absolute_error.tolist(),
        )
    ]


def print_metrics_block(name: str, metrics: dict[str, object]) -> None:
    overall = metrics["overall"]
    regions = metrics["regions"]
    print(f"{name}:")
    print(
        "  overall    "
        f"count={overall['count']:5d}  mse={overall['mse']:.6f}  mae={overall['mae']:.6f}"
    )
    for region_name in ["inside", "seen_outer", "far_outer"]:
        region = regions[region_name]
        print(
            f"  {region_name:<10} "
            f"count={region['count']:5d}  mse={region['mse']:.6f}  mae={region['mae']:.6f}"
        )


def print_probe_table(name: str, rows: list[dict[str, float]]) -> None:
    print(f"{name} probe points:")
    print("  x          sin(x)     prediction  abs_error")
    for row in rows:
        print(
            "  "
            f"{row['x']:>8.4f}   {row['target']:>8.4f}   "
            f"{row['prediction']:>10.4f}   {row['abs_error']:>9.4f}"
        )


def print_probe_summary(rows: list[dict[str, float]]) -> None:
    max_error = max(row["abs_error"] for row in rows)
    print(f"  max probe abs_error: {max_error:.6f}")
    if max_error <= 1e-4:
        print("  probe threshold met: every abs_error <= 0.0001")
    else:
        print("  probe threshold not met yet")


def print_model_summary(model: EmbeddedReLUMLP) -> None:
    learned = model.learned_parameters()
    print("  learned dynamics matrix:")
    for row in learned["dynamics"].tolist():
        print("   ", "  ".join(f"{value: .6f}" for value in row))
    print(
        "  learned initial state: "
        + ", ".join(f"{value:.6f}" for value in learned["initial_state"].tolist())
    )
    print(
        "  learned output weight: "
        + ", ".join(f"{value:.6f}" for value in learned["output_weight"].tolist())
    )
    print(f"  learned output bias: {float(learned['output_bias'].item()):.6f}")


def main() -> None:
    if EVAL_MULTIPLIER <= TRAIN_OUTSIDE_MULTIPLIER:
        raise ValueError(
            "EVAL_MULTIPLIER must be larger than TRAIN_OUTSIDE_MULTIPLIER."
        )

    set_seed(SEED)
    device = resolve_device(DEVICE)

    x_train, y_train, train_outer_radius = make_train_data(device=device)
    x_eval, y_eval, eval_outer_radius = make_eval_data(device=device)

    probe_points = torch.tensor(
        [0.0, math.pi / 2.0, 3.0 * math.pi, 8.0 * math.pi, 15.0 * math.pi],
        dtype=torch.float32,
    )

    model = EmbeddedReLUMLP(STATE_DIM).to(device)

    print("Configuration:")
    print(f"  device={device}  seed={SEED}")
    print(
        "  train points="
        f"{N_TRAIN_IN_RANGE} in-range + {N_TRAIN_OUT_OF_RANGE} out-of-range"
    )
    print(
        "  train domain="
        f"[-{train_outer_radius:.4f}, {train_outer_radius:.4f}]"
        f" with dense core [-{TRAIN_INNER_RADIUS:.4f}, {TRAIN_INNER_RADIUS:.4f}]"
    )
    print(f"  eval domain=[-{eval_outer_radius:.4f}, {eval_outer_radius:.4f}]")
    print(f"  state_dim={STATE_DIM}  steps={TRAIN_STEPS}  lr={LEARNING_RATE}")
    print(
        f"  lr schedule=warmup+cosine  warmup_fraction={LR_WARMUP_FRACTION:.2f}"
        f"  min_lr_scale={MIN_LR_SCALE:.2f}"
    )
    print(f"  optimizer=AdamW  weight_decay={WEIGHT_DECAY}")
    print()

    train_mse = train_model(model, x_train, y_train, learning_rate=LEARNING_RATE)
    metrics = evaluate_model(model, x_eval, y_eval, train_outer_radius)
    probes = evaluate_probe_points(model, probe_points, device)

    print(f"EmbeddedReLUMLP final train mse: {train_mse:.6f}")
    print_metrics_block("EmbeddedReLUMLP", metrics)
    print_probe_table("EmbeddedReLUMLP", probes)
    print_probe_summary(probes)
    print_model_summary(model)


if __name__ == "__main__":
    main()
