#!/usr/bin/env python3
"""Build the optional native C++ kernel extension."""

from __future__ import annotations

import shutil
from pathlib import Path
from setuptools import Distribution, Extension
from setuptools.command.build_ext import build_ext


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_native_cpu_cpp(*, verbose: bool = False) -> None:
    """Build an in-place `_native_cpu_cpp` extension for the current interpreter."""

    repo_root = _project_root()
    source = repo_root / "src" / "xfoil_port" / "backends" / "native_cpu_cpp.cpp"
    if not source.is_file():
        raise FileNotFoundError(f"missing source file: {source}")

    extension = Extension(
        name="xfoil_port.backends._native_cpu_cpp",
        sources=[str(source)],
    )

    dist = Distribution({"name": "xfoil-native-cpu-cpp", "ext_modules": [extension]})
    build_dir = repo_root / "build" / "xfoil_native_cpu_cpp"
    cmd = build_ext(dist)
    cmd.inplace = False
    cmd.build_lib = str(build_dir)
    cmd.build_temp = str(build_dir)
    cmd.ensure_finalized()
    if verbose:
        cmd.verbose = 1

    cmd.run()

    ext_suffixes = set(_binary_suffixes())
    so_files = []
    for candidate in Path(build_dir).rglob("_native_cpu_cpp*"):
        if candidate.is_file() and candidate.suffix == ".so":
            if any(candidate.name.endswith(ext) for ext in ext_suffixes):
                so_files.append(str(candidate))

    if not so_files:
        raise RuntimeError(f"could not locate built extension in {build_dir}")

    target_dir = repo_root / "src" / "xfoil_port" / "backends"
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(so_files[0], target_dir / Path(so_files[0]).name)


def _binary_suffixes() -> list[str]:
    from sysconfig import get_config_var

    ext_suffix = get_config_var("EXT_SUFFIX") or ".so"
    shared_object = ".so"
    return list({ext_suffix, shared_object})


def main() -> int:
    build_native_cpu_cpp(verbose=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
