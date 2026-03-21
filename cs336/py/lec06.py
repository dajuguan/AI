import math
import time
from pathlib import Path
from pprint import pformat

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    from .torch_util import get_device
except ImportError:
    from torch_util import get_device


SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
SCRIPT_DIR = Path(__file__).resolve().parent


def write_artifact(path: Path, content) -> None:
    if isinstance(content, bytes):
        path.write_bytes(content)
        return
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
        return
    if isinstance(content, list):
        path.write_text("\n".join(str(item) for item in content), encoding="utf-8")
        return
    path.write_text(pformat(content), encoding="utf-8")


def get_triton_compiled_kernel(x: torch.Tensor):
    y = torch.empty_like(x)
    return triton_gelu_kernel.warmup(
        x,
        y,
        x.numel(),
        BLOCK_SIZE=1024,
        SQRT_2_OVER_PI=SQRT_2_OVER_PI,
        grid=(triton.cdiv(x.numel(), 1024),),
    )


def print_ptx(x: torch.Tensor) -> str:
    compiled = get_triton_compiled_kernel(x)
    return compiled.asm["ptx"]


@triton.jit
def triton_gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SQRT_2_OVER_PI: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x_cubed)
    tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
    y = 0.5 * x * (1.0 + tanh_inner)

    tl.store(y_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError("triton_gelu requires a CUDA tensor")

    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    triton_gelu_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=1024,
        SQRT_2_OVER_PI=SQRT_2_OVER_PI,
    )
    return y


def pytorch_gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor) -> torch.Tensor:
    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1.0 + torch.tanh(inner))


class ManualGELUModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return manual_gelu(x)


def build_compiled_manual_gelu():
    if not hasattr(torch, "compile"):
        return None, "torch.compile is unavailable in this PyTorch build"
    try:
        return torch.compile(manual_gelu), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def benchmark(
    fn,
    x: torch.Tensor,
    *,
    warmup_iters: int = 2,
    benchmark_iters: int = 10,
) -> tuple[float, torch.Tensor]:
    for _ in range(warmup_iters):
        out = fn(x)

    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)

    start = time.perf_counter()
    for _ in range(benchmark_iters):
        out = fn(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    elapsed_s = time.perf_counter() - start

    return elapsed_s * 1e3 / benchmark_iters, out


def export_triton_compile_artifacts(
    x: torch.Tensor, output_dir: str = "artifacts/triton_gelu_compile"
) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    write_artifact(output_path / "triton_gelu_kernel.py", triton_gelu_kernel.raw_src)

    manifest: dict[str, object] = {"output_dir": str(output_path), "device": str(x.device)}

    if x.device.type != "cuda":
        (output_path / "README.txt").write_text(
            "Triton compiled artifacts require a CUDA device.\n", encoding="utf-8"
        )
        manifest["status"] = "skipped_non_cuda"
        manifest["wrote_files"] = ["triton_gelu_kernel.py", "README.txt", "manifest.txt"]
        (output_path / "manifest.txt").write_text(pformat(manifest), encoding="utf-8")
        return output_path

    try:
        compiled = get_triton_compiled_kernel(x)
        asm = getattr(compiled, "asm", None)
        manifest["status"] = "ok"
        manifest["compiled_type"] = type(compiled).__name__
        manifest["asm_keys"] = sorted(asm.keys()) if isinstance(asm, dict) else []
        wrote_files = ["triton_gelu_kernel.py"]
        if isinstance(asm, dict):
            for name in ["ttir", "ttgir", "llir", "ptx"]:
                if name in asm:
                    artifact_path = output_path / f"triton_gelu_kernel.{name}"
                    write_artifact(artifact_path, asm[name])
                    wrote_files.append(artifact_path.name)

        metadata = getattr(compiled, "metadata", None)
        if metadata is not None:
            write_artifact(output_path / "metadata.txt", metadata)
            wrote_files.append("metadata.txt")
        manifest["wrote_files"] = wrote_files + ["manifest.txt"]
    except Exception as exc:
        manifest["status"] = "compile_error"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (output_path / "error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        manifest["wrote_files"] = ["triton_gelu_kernel.py", "error.txt", "manifest.txt"]

    (output_path / "manifest.txt").write_text(pformat(manifest), encoding="utf-8")

    return output_path


def export_torch_compile_artifacts(
    x: torch.Tensor, output_dir: str = "artifacts/torch_compile_manual_gelu"
) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    manual_gelu_module = ManualGELUModule()
    exported_program = torch.export.export(manual_gelu_module, (x,))
    (output_path / "manual_gelu_exported_program.txt").write_text(
        str(exported_program), encoding="utf-8"
    )

    if not hasattr(torch, "compile"):
        (output_path / "README.txt").write_text(
            "torch.compile is unavailable in this PyTorch build.\n",
            encoding="utf-8",
        )
        return output_path

    trace_dir = output_path / "inductor_trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_config = getattr(torch._inductor.config, "trace", None)
    old_trace_enabled = getattr(trace_config, "enabled", None)
    old_debug_dir = getattr(trace_config, "debug_dir", None)

    try:
        if trace_config is not None:
            trace_config.enabled = True
            trace_config.debug_dir = str(trace_dir)
        compiled_manual_gelu = torch.compile(manual_gelu_module)
        compiled_manual_gelu(x)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
    finally:
        if trace_config is not None and old_trace_enabled is not None:
            trace_config.enabled = old_trace_enabled
        if trace_config is not None and old_debug_dir is not None:
            trace_config.debug_dir = old_debug_dir

    return output_path


def main():
    device = get_device()
    n_elements = 1 << 27
    x = torch.randn(n_elements, device=device, dtype=torch.float32)
    compiled_manual_gelu, compile_error = build_compiled_manual_gelu()

    benchmarks = [
        ("pytorch_gelu", pytorch_gelu),
        ("manual_gelu", manual_gelu),
    ]
    if compiled_manual_gelu is not None:
        benchmarks.append(("torch_compile_manual_gelu", compiled_manual_gelu))
    if device.type == "cuda":
        benchmarks.insert(0, ("triton_gelu_kernel", triton_gelu))

    print(f"device: {device}")
    print(f"input shape: {tuple(x.shape)}")
    print(f"dtype: {x.dtype}")
    print(f"warmup iters: 25")
    print(f"benchmark iters: 100")
    if compile_error is not None:
        print(f"torch_compile_manual_gelu: skipped ({compile_error})")

    export_root = SCRIPT_DIR / "artifacts"
    export_root.mkdir(exist_ok=True)
    try:
        triton_export_dir = export_triton_compile_artifacts(
            x, output_dir=export_root / "triton_gelu_compile"
        )
        print(f"triton artifacts: {triton_export_dir}")
    except Exception as exc:
        print(f"triton artifacts: skipped ({type(exc).__name__}: {exc})")

    try:
        torch_compile_export_dir = export_torch_compile_artifacts(
            x, output_dir=export_root / "torch_compile_manual_gelu"
        )
        print(f"torch.compile artifacts: {torch_compile_export_dir}")
    except Exception as exc:
        print(f"torch.compile artifacts: skipped ({type(exc).__name__}: {exc})")

    reference = pytorch_gelu(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("\nCorrectness check against pytorch_gelu:")
    for name, fn in benchmarks:
        out = fn(x)
        max_diff = (out - reference).abs().max().item()
        print(f"{name:>20}: max_abs_diff={max_diff:.6e}")

    print("\nBenchmark:")
    for name, fn in benchmarks:
        avg_ms, _ = benchmark(fn, x)
        print(f"{name:>20}: {avg_ms:8.3f} ms")


if __name__ == "__main__":
    main()
