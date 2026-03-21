# cs336/py

Python experiments for the `cs336` workspace. This directory is managed with `uv`.

## Prerequisites

- Python `3.14.3` available through `pyenv`
- `uv` installed
- NVIDIA GPU + CUDA driver if you want to run the Triton benchmark on GPU

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

From this directory:

```bash
cd /home/po/now/ai/cs336/py
uv venv
uv sync
```

This creates a local virtual environment in `.venv` and installs the dependencies from [`pyproject.toml`](/home/po/now/ai/cs336/py/pyproject.toml).

## Run

Run the GELU benchmark:

```bash
cd /home/po/now/ai/cs336/py
uv run lec06.py
```

## VS Code

Set the Python interpreter to:

```text
/home/po/now/ai/cs336/py/.venv/bin/python
```

## Notes

- `torch` is pinned to the CUDA 12.8 wheel index in [`pyproject.toml`](/home/po/now/ai/cs336/py/pyproject.toml).
- If `torch_compile_manual_gelu` is skipped, your Python build may be missing `_bz2`.
- If Triton runs on CPU only, check that `torch.cuda.is_available()` is `True` in the selected interpreter.
