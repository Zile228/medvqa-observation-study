"""Install the measured inference environment before importing GPU libraries."""

import importlib.metadata as metadata
import subprocess
import sys


def install_measured_environment(data, answerer):
    """Install packages that differ from the recorded GPU environment.

    Args:
        data: Input document containing baseline package pins.
        answerer: Q or L, selecting the package versions and CUDA wheel index.

    Raises:
        subprocess.CalledProcessError: A pip installation command fails.
    """
    pins = data["baselines"][answerer]["package_pins"]

    def installed(name):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    target_torch = pins["torch"]
    if installed("torch") != target_torch:
        cuda_tag = target_torch.split("+")[1]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "torch==" + target_torch,
                "torchvision",
                "--index-url",
                "https://download.pytorch.org/whl/" + cuda_tag,
            ],
            check=True,
        )
    packages = {k: v for k, v in pins.items() if k != "torch"}
    packages.update(
        {"numpy": "2.1.3", "huggingface-hub": "0.36.2", "sentencepiece": "0.2.2"}
    )
    changed = [
        name + "==" + version
        for name, version in packages.items()
        if installed(name) != version
    ]
    if changed:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *changed], check=True
        )
