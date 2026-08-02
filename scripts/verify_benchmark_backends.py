"""Verify the complete CPU benchmark runtime used by the devcontainer."""

from __future__ import annotations

import importlib
import platform
from importlib.metadata import version


def main() -> None:
    """Import every optional backend and report deterministic runtime facts."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("the benchmark devcontainer must run on Linux x86_64")

    packages = {
        "benchmatrix": "benchmatrix",
        "faiss-cpu": "faiss",
        "scikit-learn": "sklearn",
        "scipy": "scipy",
        "torch": "torch",
    }
    for package, module_name in packages.items():
        _ = importlib.import_module(module_name)
        print(f"{package}=={version(package)}")

    torch = importlib.import_module("torch")
    if bool(torch.cuda.is_available()):
        raise RuntimeError("the benchmark devcontainer must use CPU-only PyTorch")
    print(f"platform={platform.system().lower()}-{platform.machine()}")
    print("torch_device=cpu")


if __name__ == "__main__":
    main()
