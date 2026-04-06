"""
A simple sin(x) function to demonstrate that NN can actually learn the underlying structure instead of memorizing, to extrapolate beyond the training range, if the right architecture is used.
"""

import math
import random

import torch
import torch.nn.functional as F


N_TRAIN_IN_RANGE = 2000
N_TRAIN_OUT_OF_RANGE = 20
TRAIN_INNER_RADIUS = math.pi
TRAIN_OUTSIDE_MULTIPLIER = 10.0
EVAL_MULTIPLIER = 20.0
N_EVAL_SAMPLES = 5000
TRAIN_STEPS = 3000
LEARNING_RATE = 3e-3
LR_WARMUP_FRACTION = 0.05
MIN_LR_SCALE = 0.05
WEIGHT_DECAY = 1e-5
STATE_DIM = 2
SWIGLU_GATE_BIAS = 1.0
FOURIER_NUM_FREQS = 2
FOURIER_INITIAL_FREQUENCIES = (0.5, -0.5)
FOURIER_MODEL_DIM = 8
FOURIER_FFN_DIM = 16
FOURIER_NUM_LAYERS = 1
FOURIER_TRAIN_STEPS = 3000
FOURIER_LEARNING_RATE = 3e-3
FOURIER_FREQUENCY_LR = 1e-3
FOURIER_FREQUENCY_REG = 0.0
FOURIER_WEIGHT_DECAY = 1e-6
FOURIER_GRAD_CLIP_NORM = 0.2
FOURIER_WARMUP_FRACTION = 0.05
FOURIER_MIN_LR_SCALE = 0.10
SEED = 0
DATA_DEVICE = "cpu"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


class SwiGLUBlock(torch.nn.Module):
    def __init__(self, in_dim: int, ffn_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim < 1 or ffn_dim < 1 or out_dim < 1:
            raise ValueError("SwiGLU dimensions must be positive.")
        self.value_proj = torch.nn.Linear(in_dim, ffn_dim)
        self.gate_proj = torch.nn.Linear(in_dim, ffn_dim)
        self.out_proj = torch.nn.Linear(ffn_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.value_proj(x) * F.silu(self.gate_proj(x)))


class FourierFeatureEncoder(torch.nn.Module):
    def __init__(
        self,
        num_frequencies: int = FOURIER_NUM_FREQS,
    ) -> None:
        super().__init__()
        initial_frequencies = torch.tensor(
            FOURIER_INITIAL_FREQUENCIES,
            dtype=torch.float32,
        )
        if num_frequencies != int(initial_frequencies.numel()):
            raise ValueError(
                "FOURIER_NUM_FREQS must match the number of FOURIER_INITIAL_FREQUENCIES."
            )
        self.register_buffer("initial_frequencies", initial_frequencies)
        self.frequency_params = torch.nn.Parameter(initial_frequencies.clone())

    @property
    def output_dim(self) -> int:
        return 2 * int(self.frequency_params.numel())

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.frequency_params.copy_(self.initial_frequencies)

    def frequencies(self) -> torch.Tensor:
        return self.frequency_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = x * self.frequencies().view(1, -1)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class FourierFeatureSwiGLUMLP(torch.nn.Module):
    def __init__(
        self,
        num_frequencies: int = FOURIER_NUM_FREQS,
        model_dim: int = FOURIER_MODEL_DIM,
        ffn_dim: int = FOURIER_FFN_DIM,
        num_layers: int = FOURIER_NUM_LAYERS,
    ) -> None:
        super().__init__()
        if model_dim < 1:
            raise ValueError("FOURIER_MODEL_DIM must be positive.")
        if ffn_dim < 1:
            raise ValueError("FOURIER_FFN_DIM must be positive.")
        if num_layers < 1:
            raise ValueError("FOURIER_NUM_LAYERS must be positive.")

        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.num_layers = num_layers
        self.encoder = FourierFeatureEncoder(
            num_frequencies=num_frequencies,
        )
        self.linear_skip = torch.nn.Linear(self.encoder.output_dim, 1)
        self.input_block = SwiGLUBlock(self.encoder.output_dim, ffn_dim, model_dim)
        self.hidden_blocks = torch.nn.ModuleList(
            [SwiGLUBlock(model_dim, ffn_dim, model_dim) for _ in range(num_layers - 1)]
        )
        self.output_head = torch.nn.Linear(model_dim, 1)
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        torch.nn.init.zeros_(self.linear_skip.weight)
        torch.nn.init.zeros_(self.linear_skip.bias)
        blocks = [self.input_block, *self.hidden_blocks]
        for block in blocks:
            for layer in [block.value_proj, block.gate_proj, block.out_proj]:
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
            torch.nn.init.constant_(block.gate_proj.bias, SWIGLU_GATE_BIAS)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)
        self.residual_scale.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        hidden = self.input_block(encoded)
        for block in self.hidden_blocks:
            hidden = block(hidden)
        return self.linear_skip(encoded) + self.residual_scale * self.output_head(
            hidden
        )

    def optimizer_parameter_groups(
        self,
        learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        frequency_param = self.encoder.frequency_params
        other_params = [
            param for param in self.parameters() if param is not frequency_param
        ]
        return [
            {
                "params": other_params,
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": [frequency_param],
                "lr": FOURIER_FREQUENCY_LR,
                "weight_decay": 0.0,
            },
        ]

    def regularization_loss(self) -> torch.Tensor:
        return FOURIER_FREQUENCY_REG * (
            (self.encoder.frequency_params - self.encoder.initial_frequencies)
            .square()
            .mean()
        )

    def learned_parameters(self) -> dict[str, torch.Tensor | int]:
        return {
            "num_frequencies": int(self.encoder.frequency_params.numel()),
            "frequencies": self.encoder.frequencies().detach().cpu(),
            "model_dim": self.model_dim,
            "ffn_dim": self.ffn_dim,
            "num_layers": self.num_layers,
            "input_value_norm": self.input_block.value_proj.weight.detach()
            .cpu()
            .norm(),
            "input_gate_norm": self.input_block.gate_proj.weight.detach().cpu().norm(),
            "hidden_out_norms": torch.tensor(
                [
                    block.out_proj.weight.detach().cpu().norm().item()
                    for block in self.hidden_blocks
                ]
            ),
            "linear_skip_weight": self.linear_skip.weight.detach().cpu().squeeze(0),
            "linear_skip_bias": self.linear_skip.bias.detach().cpu(),
            "residual_scale": self.residual_scale.detach().cpu(),
            "output_weight_norm": self.output_head.weight.detach().cpu().norm(),
            "output_bias": self.output_head.bias.detach().cpu(),
        }


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
    device: str | torch.device = DATA_DEVICE,
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
    device: str | torch.device = DATA_DEVICE,
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
    weight_decay: float = WEIGHT_DECAY,
    warmup_fraction: float = LR_WARMUP_FRACTION,
    min_lr_scale: float = MIN_LR_SCALE,
    grad_clip_norm: float | None = None,
) -> float:
    if train_steps <= 0:
        raise ValueError("TRAIN_STEPS must be positive.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1).")
    if not 0.0 < min_lr_scale <= 1.0:
        raise ValueError("min_lr_scale must be in (0, 1].")
    if grad_clip_norm is not None and grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive when provided.")

    optimizer_parameter_groups = None
    if hasattr(model, "optimizer_parameter_groups"):
        optimizer_parameter_groups = model.optimizer_parameter_groups(
            learning_rate,
            weight_decay,
        )

    if optimizer_parameter_groups is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_parameter_groups)

    warmup_steps = int(train_steps * warmup_fraction)

    def get_lr_for_step(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            warmup_progress = (step + 1) / warmup_steps
            return learning_rate * warmup_progress

        decay_steps = max(1, train_steps - warmup_steps)
        decay_progress = (step - warmup_steps) / decay_steps
        decay_progress = min(max(decay_progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        min_lr = learning_rate * min_lr_scale
        return min_lr + (learning_rate - min_lr) * cosine_scale

    model.train()
    for step in range(train_steps):
        current_lr = get_lr_for_step(step)
        for param_group in optimizer.param_groups:
            if "base_lr" not in param_group:
                param_group["base_lr"] = param_group["lr"]
            base_lr = param_group["base_lr"]
            scaled_lr = current_lr
            if learning_rate > 0.0:
                scaled_lr = current_lr * (base_lr / learning_rate)
            param_group["lr"] = scaled_lr
        prediction = model(x_train)
        loss = torch.mean((prediction - y_train) ** 2)
        if hasattr(model, "regularization_loss"):
            loss = loss + model.regularization_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_frequency_values(values: torch.Tensor) -> str:
    return ", ".join(f"{value:.6f}" for value in values.tolist())


def print_matrix_exp_model_summary(model: EmbeddedReLUMLP) -> None:
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


def print_fourier_model_summary(model: FourierFeatureSwiGLUMLP) -> None:
    learned = model.learned_parameters()
    print(
        "  Fourier encoder: "
        f"num_frequencies={learned['num_frequencies']}  "
        + ", ".join(f"{value:.3f}" for value in learned["frequencies"].tolist())
    )
    print(
        "  SwiGLU stack: "
        f"model_dim={learned['model_dim']}  "
        f"ffn_dim={learned['ffn_dim']}  "
        f"layers={learned['num_layers']}"
    )
    print(
        "  input block norms: "
        f"value={float(learned['input_value_norm']):.6f}  "
        f"gate={float(learned['input_gate_norm']):.6f}"
    )
    print(
        "  linear skip: "
        + ", ".join(f"{value:.6f}" for value in learned["linear_skip_weight"].tolist())
        + f"  bias={float(learned['linear_skip_bias'].item()):.6f}"
    )
    if learned["hidden_out_norms"].numel() > 0:
        print(
            "  hidden out-proj norms: "
            + ", ".join(
                f"{value:.6f}" for value in learned["hidden_out_norms"].tolist()
            )
        )
    print(f"  residual_scale={float(learned['residual_scale'].item()):.6f}")
    print(
        "  learned output head: "
        f"weight_norm={float(learned['output_weight_norm']):.6f}  "
        f"bias={float(learned['output_bias'].item()):.6f}"
    )


def run_experiment(
    name: str,
    model: torch.nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    train_outer_radius: float,
    probe_points: torch.Tensor,
    device: torch.device,
    train_steps: int = TRAIN_STEPS,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_fraction: float = LR_WARMUP_FRACTION,
    min_lr_scale: float = MIN_LR_SCALE,
    grad_clip_norm: float | None = None,
) -> None:
    train_mse = train_model(
        model,
        x_train,
        y_train,
        train_steps=train_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_fraction=warmup_fraction,
        min_lr_scale=min_lr_scale,
        grad_clip_norm=grad_clip_norm,
    )
    metrics = evaluate_model(model, x_eval, y_eval, train_outer_radius)
    probes = evaluate_probe_points(model, probe_points, device)

    print(f"{name} parameter_count: {count_parameters(model)}")
    print(f"{name} final train mse: {train_mse:.6f}")
    print_metrics_block(name, metrics)
    print_probe_table(name, probes)
    print_probe_summary(probes)
    if isinstance(model, EmbeddedReLUMLP):
        print_matrix_exp_model_summary(model)
    elif isinstance(model, FourierFeatureSwiGLUMLP):
        print(
            f"{name} initial frequencies: "
            + format_frequency_values(model.encoder.initial_frequencies.detach().cpu())
        )
        print(
            f"{name} trained frequencies: "
            + format_frequency_values(model.encoder.frequencies().detach().cpu())
        )
        print_fourier_model_summary(model)
    print()


def main() -> None:
    if EVAL_MULTIPLIER <= TRAIN_OUTSIDE_MULTIPLIER:
        raise ValueError(
            "EVAL_MULTIPLIER must be larger than TRAIN_OUTSIDE_MULTIPLIER."
        )

    set_seed(SEED)
    data_device = resolve_device(DATA_DEVICE)
    device = resolve_device(DEVICE)

    x_train_cpu, y_train_cpu, train_outer_radius = make_train_data(device=data_device)
    x_eval_cpu, y_eval_cpu, eval_outer_radius = make_eval_data(device=data_device)

    probe_points = torch.tensor(
        [0.0, math.pi / 2.0, 3.0 * math.pi, 8.0 * math.pi, 15.0 * math.pi],
        dtype=torch.float32,
    )

    x_train = x_train_cpu.to(device)
    y_train = y_train_cpu.to(device)
    x_eval = x_eval_cpu.to(device)
    y_eval = y_eval_cpu.to(device)

    set_seed(SEED)
    matrix_exp_model = EmbeddedReLUMLP(STATE_DIM).to(device)

    set_seed(SEED)
    fourier_model = FourierFeatureSwiGLUMLP(
        num_frequencies=FOURIER_NUM_FREQS,
        model_dim=FOURIER_MODEL_DIM,
        ffn_dim=FOURIER_FFN_DIM,
        num_layers=FOURIER_NUM_LAYERS,
    ).to(device)

    print("Configuration:")
    print(f"  data_device={data_device}" f"  device={device}" f"  seed={SEED}")
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
    print(
        f"  fourier num_frequencies={FOURIER_NUM_FREQS}"
        f"  initial_frequencies={format_frequency_values(torch.tensor(FOURIER_INITIAL_FREQUENCIES))}"
        f"  model_dim={FOURIER_MODEL_DIM}"
        f"  ffn_dim={FOURIER_FFN_DIM}"
        f"  layers={FOURIER_NUM_LAYERS}"
    )
    print(
        f"  fourier_steps={FOURIER_TRAIN_STEPS}"
        f"  fourier_lr={FOURIER_LEARNING_RATE}"
        f"  fourier_freq_lr={FOURIER_FREQUENCY_LR}"
        f"  fourier_freq_reg={FOURIER_FREQUENCY_REG}"
        f"  fourier_wd={FOURIER_WEIGHT_DECAY}"
        f"  fourier_grad_clip={FOURIER_GRAD_CLIP_NORM}"
        f"  fourier_warmup_fraction={FOURIER_WARMUP_FRACTION:.2f}"
        f"  fourier_min_lr_scale={FOURIER_MIN_LR_SCALE:.2f}"
    )
    print(f"  optimizer=AdamW  weight_decay={WEIGHT_DECAY}")
    print()

    run_experiment(
        "EmbeddedReLUMLP",
        matrix_exp_model,
        x_train,
        y_train,
        x_eval,
        y_eval,
        train_outer_radius,
        probe_points,
        device,
    )
    run_experiment(
        "FourierFeatureSwiGLUMLP",
        fourier_model,
        x_train,
        y_train,
        x_eval,
        y_eval,
        train_outer_radius,
        probe_points,
        device,
        train_steps=FOURIER_TRAIN_STEPS,
        learning_rate=FOURIER_LEARNING_RATE,
        weight_decay=FOURIER_WEIGHT_DECAY,
        warmup_fraction=FOURIER_WARMUP_FRACTION,
        min_lr_scale=FOURIER_MIN_LR_SCALE,
        grad_clip_norm=FOURIER_GRAD_CLIP_NORM,
    )


if __name__ == "__main__":
    main()
