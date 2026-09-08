"""Data pipeline: STL-10 loading, the 80k/10k/10k split, seeding and the patch generators.

The pipeline reproduces the course assignment exactly. Every 96x96 image is cut into a 3x3 grid of
32x32 cells, each cell is centre-cropped to 28x28 (a 2-pixel border is discarded on every side, so
neighbouring patches never share an edge), and the 9 patches are returned in a random order
together with the full image as the target.
"""
from __future__ import annotations

import os

import keras
import numpy as np
from keras.utils import PyDataset

STL10_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
N_IMAGES = 100_000
IMAGE_SIZE = 96
GRID = 3
PATCH_SIZE = 32                          # grid cell
CROP_SIZE = 28                           # kept centre of each cell
MARGIN = (PATCH_SIZE - CROP_SIZE) // 2   # eroded border on each side (2 px)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and the Keras backend (weight init, data order, patch permutations)."""
    keras.utils.set_random_seed(seed)


def load_stl10(cache_dir: str | None = None) -> np.ndarray:
    """Download STL-10 once (2.6 GB) and return the unlabeled split as uint8 ``(100000, 96, 96, 3)``.

    STL-10 stores each image channel-first and column-major, hence the ``(0, 3, 2, 1)`` transpose
    (this is the official loader's recipe). The archive is cached by Keras under ``~/.keras/datasets``.
    """
    path = keras.utils.get_file("stl10_binary.tar.gz", origin=STL10_URL, extract=True, cache_dir=cache_dir)
    base_dir = os.path.dirname(path)
    candidates = [
        os.path.join(base_dir, "stl10_binary_extracted", "stl10_binary", "unlabeled_X.bin"),  # Keras 3
        os.path.join(base_dir, "stl10_binary", "unlabeled_X.bin"),                            # Keras 2
    ]
    expected = N_IMAGES * 3 * IMAGE_SIZE * IMAGE_SIZE
    for filepath in candidates:
        if os.path.exists(filepath) and os.path.getsize(filepath) == expected:
            break
    else:
        raise FileNotFoundError(f"unlabeled_X.bin missing or incomplete; looked in {candidates}")
    data = np.fromfile(filepath, dtype=np.uint8)
    images = np.reshape(data, (-1, 3, IMAGE_SIZE, IMAGE_SIZE))   # (N, C, W, H): column-major storage
    return np.transpose(images, (0, 3, 2, 1))                    # (N, H, W, C)


def split(images: np.ndarray, n_train: int = 80_000, n_val: int = 10_000):
    """Contiguous, disjoint train / validation / test slices (default 80k / 10k / 10k)."""
    return images[:n_train], images[n_train:n_train + n_val], images[n_train + n_val:]


def known_pixel_mask() -> np.ndarray:
    """``(96, 96)`` float32 mask of the pixels covered by the 9 centre crops (1 = known, 0 = seam)."""
    m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.float32)
    for r in range(GRID):
        for c in range(GRID):
            m[r * PATCH_SIZE + MARGIN: r * PATCH_SIZE + MARGIN + CROP_SIZE,
              c * PATCH_SIZE + MARGIN: c * PATCH_SIZE + MARGIN + CROP_SIZE] = 1.0
    return m


class PatchGenerator(PyDataset):
    """The assignment's generator: ``(B, 9, 28, 28, 3)`` shuffled patches -> ``(B, 96, 96, 3)`` image.

    Patch index ``3 * row + col`` is the grid position. Permutations are drawn from NumPy's global RNG
    on every call, so batches are reproducible only if :func:`set_seed` was called; use
    :class:`SeededPatchGenerator` for evaluation.
    """

    def __init__(self, images, batch_size=32, patch_size=PATCH_SIZE, crop_size=CROP_SIZE, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.images = images
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def _cells(self, img: np.ndarray) -> list[np.ndarray]:
        """The 9 centre-cropped cells of one float image, in grid order."""
        ps, cs, m = self.patch_size, self.crop_size, (self.patch_size - self.crop_size) // 2
        return [img[r * ps + m: r * ps + m + cs, c * ps + m: c * ps + m + cs, :]
                for r in range(GRID) for c in range(GRID)]

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        n = len(batch_indices)
        X = np.zeros((n, GRID * GRID, self.crop_size, self.crop_size, 3), dtype="float32")
        Y = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32")
        for i, img_idx in enumerate(batch_indices):
            full_img = self.images[img_idx].astype("float32") / 255.0
            Y[i] = full_img
            patches = self._cells(full_img)
            order = np.random.permutation(GRID * GRID)         # order[slot] = original grid position
            for slot_idx, original_pos in enumerate(order):
                X[i, slot_idx] = patches[original_pos]
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


class SeededPatchGenerator(PatchGenerator):
    """:class:`PatchGenerator` with a deterministic per-batch RNG and access to the true permutation.

    ``__getitem__`` returns exactly what :class:`PatchGenerator` returns; :meth:`get_batch_with_order`
    also returns ``order`` with ``order[slot]`` = original grid position (0-8) of the patch in ``slot``.
    """

    def __init__(self, images, batch_size=64, seed=0, **kwargs):
        super().__init__(images, batch_size=batch_size, shuffle=False, **kwargs)
        self.seed = seed

    def get_batch_with_order(self, idx):
        rng = np.random.RandomState(self.seed * 1_000_003 + idx)
        batch_indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        n = len(batch_indices)
        X = np.zeros((n, GRID * GRID, self.crop_size, self.crop_size, 3), dtype="float32")
        Y = np.zeros((n, IMAGE_SIZE, IMAGE_SIZE, 3), dtype="float32")
        order = np.zeros((n, GRID * GRID), dtype=np.int32)
        for i, img_idx in enumerate(batch_indices):
            full_img = self.images[img_idx].astype("float32") / 255.0
            Y[i] = full_img
            patches = self._cells(full_img)
            order[i] = rng.permutation(GRID * GRID)
            for slot_idx, original_pos in enumerate(order[i]):
                X[i, slot_idx] = patches[original_pos]
        return X, Y, order

    def __getitem__(self, idx):
        X, Y, _ = self.get_batch_with_order(idx)
        return X, Y
