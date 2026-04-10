"""Validate Zarr datasets for corrupt JpegXL chunks.

Usage:
  python tests/validate_zarr_integrity.py /path/to/dataset.zarr.zip

Optional:
  --array data/camera0_rgb   Zarr array to scan (default: data/camera0_rgb)
  --max-frames 0             Limit frames to scan (0 = all)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable, Optional, Tuple

import numpy as np
import zarr

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
os.chdir(str(ROOT_DIR))

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

try:
    from imagecodecs._jpegxl import JpegxlError
except Exception:  # pragma: no cover - fallback if imagecodecs internals change
    class JpegxlError(Exception):
        pass


def _iter_frame_ranges(total: int, chunk: int, max_frames: int) -> Tuple[int, int]:
    remaining = total if max_frames <= 0 else min(total, max_frames)
    start = 0
    while start < remaining:
        end = min(start + chunk, remaining)
        yield start, end
        start = end


def _scan_array(arr: zarr.Array, array_key: str, max_frames: int) -> int:
    total_frames = arr.shape[0]
    chunk = arr.chunks[0]

    failures = 0
    for start, end in _iter_frame_ranges(total_frames, chunk, max_frames):
        try:
            _ = arr[start:end]
        except JpegxlError as exc:
            failures += 1
            print(
                f"[validate_zarr_integrity] JpegXL decode failed for "
                f"frames {start}:{end} in '{array_key}'. Error: {exc}"
            )
        except Exception as exc:
            failures += 1
            print(
                f"[validate_zarr_integrity] Unexpected error for "
                f"frames {start}:{end} in '{array_key}'. Error: {exc}"
            )
    return failures


def _array_items(root: zarr.Group, prefix: str = "") -> Iterable[Tuple[str, zarr.Array]]:
    for name, array in root.arrays():
        full_name = f"{prefix}{name}" if prefix else name
        yield full_name, array
    for name, group in root.groups():
        full_prefix = f"{prefix}{name}/" if prefix else f"{name}/"
        yield from _array_items(group, full_prefix)


def validate_zarr(path: str, array_key: Optional[str], max_frames: int) -> int:
    register_codecs()
    store = zarr.ZipStore(path, mode="r")
    root = zarr.group(store=store)
    failures = 0
    if array_key is not None:
        if array_key not in root:
            raise KeyError(f"Array '{array_key}' not found in {path}")
        failures += _scan_array(root[array_key], array_key, max_frames)
    else:
        for name, array in _array_items(root):
            failures += _scan_array(array, name, max_frames)

    store.close()
    if failures == 0:
        print("[validate_zarr_integrity] OK: no corrupted chunks detected.")
    else:
        print(
            f"[validate_zarr_integrity] FAIL: {failures} corrupted chunk(s) detected."
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Zarr dataset integrity.")
    parser.add_argument("path", help="Path to dataset.zarr.zip")
    parser.add_argument(
        "--array",
        default=None,
        help="Zarr array key to scan (default: scan all arrays)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to scan (0 = all)",
    )
    args = parser.parse_args()

    failures = validate_zarr(args.path, args.array, args.max_frames)
    sys.exit(1 if failures > 0 else 0)


if __name__ == "__main__":
    main()
