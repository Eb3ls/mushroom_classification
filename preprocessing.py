"""GrabCut preprocessing and validation for the generated image cache."""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from warnings import warn

import cv2
import numpy as np
import pandas as pd


IDENTITY_COLUMNS = ["image_path", "label", "edible"]
ITERS_COLUMN = "grabcut_iters"
CROP_COLUMN = "grabcut_cropped_pixels"


class PreprocessingError(RuntimeError):
    """Raised when a preprocessing run cannot be completed safely."""


def apply_grabcut(
    img: np.ndarray,
    iters: int = 3,
    cropped_pixels: int = 10,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Segment an image, retaining the notebook's tiny-foreground fallback."""
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        rect = (
            cropped_pixels,
            cropped_pixels,
            img.shape[1] - 2 * cropped_pixels,
            img.shape[0] - 2 * cropped_pixels,
        )
        mode = cv2.GC_INIT_WITH_RECT
    else:
        mask = mask.astype(np.uint8, copy=False)
        rect = None
        mode = cv2.GC_INIT_WITH_MASK

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    mask, _, _ = cv2.grabCut(img, mask, rect, bg_model, fg_model, iters, mode)

    mask_binary = np.where(
        (mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 255
    ).astype(np.uint8)
    segmented = cv2.bitwise_and(img, img, mask=mask_binary)
    if segmented.astype(bool).sum() < img.shape[0] * img.shape[1] * 3 * 0.01:
        warn(
            "GrabCut resulted in very small foreground, returning original image"
        )
        return img
    return segmented


def _output_path(source_path: object) -> Path:
    source = Path(str(source_path))
    return source.with_name(f"{source.stem}_grabcut{source.suffix}")


def _temporary_path(path: Path, purpose: str) -> Path:
    name = f".{path.stem}.{purpose}-{uuid.uuid4().hex}{path.suffix}"
    return path.with_name(name)


def _has_required_source_columns(df: pd.DataFrame, path_col: str) -> bool:
    required = set(IDENTITY_COLUMNS) | {path_col}
    return required.issubset(df.columns) and not df[list(required)].isnull().any().any()


def _is_readable_image(path: Path) -> bool:
    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR) is not None
    except Exception:
        return False


def check_csv_done(
    df: pd.DataFrame,
    path: str | os.PathLike[str],
    iters: int = 2,
    cropped_pixels: int = 10,
    path_col: str = "image_path",
) -> bool:
    """Return whether the CSV exactly describes the requested readable cache."""
    if not isinstance(df, pd.DataFrame) or not _has_required_source_columns(
        df, path_col
    ):
        return False

    manifest_path = Path(path)
    if not manifest_path.is_file():
        return False
    try:
        existing = pd.read_csv(
            manifest_path,
            dtype={"image_path": str, "label": str, "grabcut_path": str},
        )
    except Exception:
        return False

    required = set(IDENTITY_COLUMNS) | {
        "grabcut_path",
        ITERS_COLUMN,
        CROP_COLUMN,
    }
    if not required.issubset(existing.columns) or len(existing) != len(df):
        return False
    if existing[list(required)].isnull().any().any():
        return False

    if existing["image_path"].tolist() != df["image_path"].map(str).tolist():
        return False
    if existing["label"].tolist() != df["label"].map(str).tolist():
        return False
    if existing["edible"].tolist() != df["edible"].tolist():
        return False

    outputs = [_output_path(value) for value in df[path_col]]
    if existing["grabcut_path"].tolist() != [str(output) for output in outputs]:
        return False
    if existing[ITERS_COLUMN].tolist() != [iters] * len(df):
        return False
    if existing[CROP_COLUMN].tolist() != [cropped_pixels] * len(df):
        return False
    return all(_is_readable_image(output) for output in outputs)


def grabcut_save_from_df(
    df: pd.DataFrame,
    path_col: str,
    save_path: str | os.PathLike[str] | None,
    iters: int = 2,
    cropped_pixels: int = 10,
) -> list[tuple[str, bool]]:
    """Generate images, then publish outputs and their manifest as one cache."""
    if not isinstance(df, pd.DataFrame) or not _has_required_source_columns(
        df, path_col
    ):
        raise ValueError(
            f"DataFrame must contain non-null columns {IDENTITY_COLUMNS + [path_col]}"
        )
    if save_path is None or str(save_path).strip() == "":
        save_path = "segmented_path.csv"

    manifest_path = Path(save_path)
    row_outputs = [_output_path(value) for value in df[path_col]]
    tasks: list[tuple[Path, Path, Path]] = []
    seen_outputs: set[str] = set()
    for source_value, output in zip(df[path_col], row_outputs, strict=True):
        if str(output) not in seen_outputs:
            seen_outputs.add(str(output))
            staged = _temporary_path(output, "grabcut-stage")
            tasks.append((Path(str(source_value)), output, staged))

    workers = min(32, (os.cpu_count() or 4) + 4)
    print(f"Using {workers} workers for GrabCut processing.")

    def process_one(task: tuple[Path, Path, Path]) -> None:
        source, _, staged = task
        try:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("unable to read source image")
            segmented = apply_grabcut(
                image.astype(np.uint8, copy=False),
                iters=iters,
                cropped_pixels=cropped_pixels,
            )
            if not cv2.imwrite(str(staged), segmented):
                raise OSError("failed to write segmented image")
            if not _is_readable_image(staged):
                raise OSError("written segmented image is unreadable")
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to process source image '{source}': {exc}"
            ) from exc

    staged_images = [staged for _, _, staged in tasks]
    staged_manifest = _temporary_path(manifest_path, "manifest-stage")
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(process_one, tasks))

        manifest = df.copy(deep=True)
        manifest["grabcut_path"] = [str(output) for output in row_outputs]
        manifest[ITERS_COLUMN] = iters
        manifest[CROP_COLUMN] = cropped_pixels
        try:
            manifest.to_csv(staged_manifest, index=False)
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to write cache manifest '{manifest_path}': {exc}"
            ) from exc

        # From this point, any failure must be a cache miss instead of leaving
        # stale metadata that claims partially replaced outputs are valid.
        try:
            manifest_path.unlink(missing_ok=True)
            for _, output, staged in tasks:
                os.replace(staged, output)
            os.replace(staged_manifest, manifest_path)
        except Exception as exc:
            manifest_path.unlink(missing_ok=True)
            raise PreprocessingError(
                f"Failed to publish GrabCut cache '{manifest_path}': {exc}"
            ) from exc
    finally:
        staged_manifest.unlink(missing_ok=True)
        for staged in staged_images:
            staged.unlink(missing_ok=True)

    return [(str(output), True) for output in row_outputs]
