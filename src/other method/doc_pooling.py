from __future__ import annotations

import math
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PATCH_SIZE = 14
MERGE_SIZE = 2
FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 56 * 56
MAX_PIXELS = 28 * 28 * 1280
MAX_RATIO = 200


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def pool_1d(emb: np.ndarray, keep_ratio: float) -> np.ndarray:
    n_tokens = emb.shape[0]
    n_keep = max(1, int(round(n_tokens * keep_ratio)))
    if n_keep >= n_tokens:
        return emb.copy()

    edges = np.linspace(0, n_tokens, num=n_keep + 1, dtype=int)
    out = np.empty((n_keep, emb.shape[1]), dtype=emb.dtype)
    for i in range(n_keep):
        start = edges[i]
        end = max(edges[i] + 1, edges[i + 1])
        out[i] = emb[start:end].mean(axis=0)
    return _l2_normalize(out)


def _pick_2d_block(height: int, width: int, keep_ratio: float) -> tuple[int, int]:
    n_tokens = height * width
    target = max(1, int(round(n_tokens * keep_ratio)))
    if target >= n_tokens:
        return 1, 1

    best: tuple[tuple[int, int], int, int] | None = None
    max_side = max(2, int(math.ceil(math.sqrt(n_tokens / target))) + 2)
    for block_h in range(1, max_side + 1):
        for block_w in range(1, max_side + 1):
            if block_h == 1 and block_w == 1:
                continue
            out_n = math.ceil(height / block_h) * math.ceil(width / block_w)
            score = (abs(out_n - target), abs(block_h - block_w))
            if best is None or score < best[0]:
                best = (score, block_h, block_w)

    if best is None:
        return 1, 1
    return best[1], best[2]


def pool_2d_image_block(img_emb: np.ndarray, keep_ratio: float, grid: tuple[int, int]) -> np.ndarray:
    height, width = grid
    if img_emb.shape[0] != height * width:
        raise ValueError(f"image-token count {img_emb.shape[0]} != H*W = {height * width}")
    if keep_ratio >= 1.0:
        return img_emb.copy()

    block_h, block_w = _pick_2d_block(height, width, keep_ratio)
    dim = img_emb.shape[1]
    grid_emb = img_emb.reshape(height, width, dim)
    out_h = math.ceil(height / block_h)
    out_w = math.ceil(width / block_w)
    out = np.empty((out_h, out_w, dim), dtype=img_emb.dtype)

    for i in range(out_h):
        for j in range(out_w):
            r0, r1 = i * block_h, min((i + 1) * block_h, height)
            c0, c1 = j * block_w, min((j + 1) * block_w, width)
            out[i, j] = grid_emb[r0:r1, c0:c1].reshape(-1, dim).mean(axis=0)

    return _l2_normalize(out.reshape(out_h * out_w, dim))


def locate_image_block(emb: np.ndarray, n_expected: int) -> tuple[int, int]:
    n_expected = min(n_expected, emb.shape[0])
    leftover = emb.shape[0] - n_expected
    start = leftover // 2
    return start, start + n_expected


def _fallback_grid(emb: np.ndarray, non_image_tokens: int = 4) -> tuple[int, int]:
    n_img = max(1, emb.shape[0] - non_image_tokens)
    height = int(math.isqrt(n_img))
    while height > 1 and n_img % height != 0:
        height -= 1
    width = n_img // height if height > 0 else n_img
    return height, width


def pool_2d(
    emb: np.ndarray,
    keep_ratio: float,
    grid: tuple[int, int] | None = None,
    image_slice: tuple[int, int] | None = None,
    non_image_tokens: int = 4,
) -> np.ndarray:
    if grid is None:
        grid = _fallback_grid(emb, non_image_tokens=non_image_tokens)

    n_img = grid[0] * grid[1]
    if image_slice is None:
        image_slice = locate_image_block(emb, n_img)

    start, end = image_slice
    img_emb = emb[start:end]
    if img_emb.shape[0] != n_img:
        grid = (1, img_emb.shape[0])

    pooled = pool_2d_image_block(img_emb, keep_ratio, grid)
    return np.concatenate([emb[:start], pooled, emb[end:]], axis=0)


def _round_by_factor(n: float, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by_factor(n: float, factor: int) -> int:
    return math.ceil(n / factor) * factor


def _floor_by_factor(n: float, factor: int) -> int:
    return math.floor(n / factor) * factor


def smart_resize_local(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    max_ratio: int = MAX_RATIO,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > max_ratio:
        raise ValueError(f"Aspect ratio exceeds {max_ratio}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def decode_image_field(image_field) -> Image.Image:
    from PIL import Image

    if image_field is None:
        raise ValueError("image field is None")
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB")
    if isinstance(image_field, dict):
        image_bytes = image_field.get("bytes")
        if image_bytes is None:
            raise ValueError("image dict has no 'bytes' key")
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(BytesIO(image_field)).convert("RGB")
    raise TypeError(f"Unknown image field type: {type(image_field)}")


def infer_patch_grid(
    pil_img: Any,
    patch_size: int = PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
) -> tuple[int, int]:
    width, height = pil_img.size
    h_bar, w_bar = smart_resize_local(height, width)
    return h_bar // patch_size // merge_size, w_bar // patch_size // merge_size


def infer_grids_from_pages(
    embeddings: list[np.ndarray],
    pages_parquet: str | Path | None,
    page_indices: Iterable[int] | None = None,
    non_image_tokens: int = 4,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    grids = [_fallback_grid(emb, non_image_tokens=non_image_tokens) for emb in embeddings]
    slices = [locate_image_block(emb, grids[i][0] * grids[i][1]) for i, emb in enumerate(embeddings)]

    if pages_parquet is None:
        return grids, slices

    path = Path(pages_parquet)
    if not path.exists():
        return grids, slices

    import pandas as pd

    pages_df = pd.read_parquet(path)
    if "image" not in pages_df.columns:
        return grids, slices

    indices = list(range(len(embeddings))) if page_indices is None else list(page_indices)
    for pos, page_idx in enumerate(indices):
        if pos >= len(embeddings) or page_idx >= len(pages_df):
            continue
        try:
            pil_img = decode_image_field(pages_df.iloc[int(page_idx)]["image"])
            grid = infer_patch_grid(pil_img)
        except Exception:
            continue

        n_img = grid[0] * grid[1]
        image_slice = locate_image_block(embeddings[pos], n_img)
        if image_slice[1] - image_slice[0] != n_img:
            n_img = image_slice[1] - image_slice[0]
            grid = (1, n_img)
        grids[pos] = grid
        slices[pos] = image_slice

    return grids, slices


def compress_embeddings(
    embeddings: list[np.ndarray],
    strategy: str,
    keep_ratio: float,
    grids: list[tuple[int, int]] | None = None,
    slices: list[tuple[int, int]] | None = None,
    non_image_tokens: int = 4,
) -> list[np.ndarray]:
    if strategy == "pool1d":
        return [pool_1d(emb, keep_ratio) for emb in embeddings]
    if strategy != "pool2d":
        raise ValueError(f"Unknown pooling strategy: {strategy}")

    return [
        pool_2d(
            emb,
            keep_ratio,
            grid=None if grids is None else grids[i],
            image_slice=None if slices is None else slices[i],
            non_image_tokens=non_image_tokens,
        )
        for i, emb in enumerate(embeddings)
    ]
