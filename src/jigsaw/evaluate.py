"""Seeded evaluation on the test split.

Reports the per-image MAE (mean and standard deviation over images), the routing accuracy computed
from the true permutation, the assignment's mean-patch baseline, and draws the error-map figure.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from .data import CROP_SIZE, GRID, IMAGE_SIZE, SeededPatchGenerator
from .trainer import count_trainable


def mean_patch_baseline(X: np.ndarray) -> np.ndarray:
    """The assignment's baseline: every cell is the mean of the 9 patches, tiled to 84x84 and resized to 96x96."""
    B = tf.shape(X)[0]
    mean_patch = tf.reduce_mean(X, axis=1)                                        # (B, 28, 28, 3)
    tiled = tf.repeat(mean_patch[:, None], repeats=GRID * GRID, axis=1)          # (B, 9, 28, 28, 3)
    out = tf.reshape(tiled, (B, GRID, GRID, CROP_SIZE, CROP_SIZE, 3))
    out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
    out = tf.reshape(out, (B, GRID * CROP_SIZE, GRID * CROP_SIZE, 3))
    return tf.image.resize(out, (IMAGE_SIZE, IMAGE_SIZE)).numpy()


def evaluate(model, images, batch_size: int = 64, seed: int = 0, max_images: int | None = None) -> dict:
    """Evaluate ``model`` on ``images`` with a seeded generator (exactly reproducible).

    Returns a dict with ``mae``/``std`` (per-image MAE mean and std), ``baseline_mae``/``baseline_std``,
    ``exact_routing`` (patch sent to its true cell), ``content_routing`` (patch sent to a cell whose true
    content is pixel-identical to it), ``solved`` (all 9 patches exact), ``params`` and the per-image arrays.
    """
    if max_images:
        images = images[:max_images]
    gen = SeededPatchGenerator(images, batch_size=batch_size, seed=seed)
    per_image, base_per_image, exact_ok, content_ok = [], [], [], []
    for i in range(len(gen)):
        X, Y, order = gen.get_batch_with_order(i)
        P, _, pred = model.forward(X, model.eval_tau, training=False)          # one forward pass
        P, pred = P.numpy(), pred.numpy()
        per_image.append(np.abs(pred - Y).mean(axis=(1, 2, 3)))                # one value per image
        base_per_image.append(np.abs(mean_patch_baseline(X) - Y).mean(axis=(1, 2, 3)))
        pred_pos = P.argmax(axis=2)                                             # cell chosen for each slot
        exact_ok.append(pred_pos == order)
        cells = model.extract_cells(Y).numpy()                                  # (n, 9, 28, 28, 3), grid order
        placed = np.take_along_axis(cells, pred_pos[:, :, None, None, None], axis=1)
        content_ok.append(np.abs(placed - X).mean(axis=(2, 3, 4)) == 0)         # exact crops -> 0 iff right
    per_image, base_per_image = np.concatenate(per_image), np.concatenate(base_per_image)
    exact_ok, content_ok = np.concatenate(exact_ok), np.concatenate(content_ok)
    assert per_image.shape == (len(images),)
    return dict(
        n_images=int(len(images)),
        mae=float(per_image.mean()), std=float(per_image.std()),
        baseline_mae=float(base_per_image.mean()), baseline_std=float(base_per_image.std()),
        exact_routing=float(exact_ok.mean()), content_routing=float(content_ok.mean()),
        solved=float(exact_ok.all(axis=1).mean()),
        params=count_trainable(model),
        per_image_mae=per_image, baseline_per_image_mae=base_per_image,
    )


def metrics_table(res: dict) -> str:
    """Markdown table: mean-patch baseline vs. ours."""
    rows = [
        ("Test MAE (mean over images)", f"{res['baseline_mae']:.4f}", f"**{res['mae']:.4f}**"),
        ("Std of per-image MAE", f"{res['baseline_std']:.4f}", f"{res['std']:.4f}"),
        ("Routing accuracy (exact permutation)", "-", f"{100 * res['exact_routing']:.1f}%"),
        ("Routing accuracy (content-equivalent)", "-", f"{100 * res['content_routing']:.1f}%"),
        ("Puzzles fully solved", "-", f"{100 * res['solved']:.1f}%"),
        ("Trainable parameters", "0", f"{res['params']:,}"),
    ]
    lines = [f"| Metric ({res['n_images']:,} test images) | Mean-patch baseline | Ours |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def save_error_maps(model, images, path: str, rows: int = 6, seed: int = 1, vmax: float = 0.5) -> None:
    """Figure with ``rows`` test samples: shuffled input | ground truth | reconstruction | abs-error map."""
    import matplotlib.pyplot as plt

    X, GT, _ = SeededPatchGenerator(images, batch_size=rows, seed=seed).get_batch_with_order(0)
    PR = np.clip(model(X, training=False).numpy(), 0, 1)

    def tile(patches):                                   # 9 patches (slot order) -> 84x84 grid
        r = [np.concatenate([patches[i * GRID + j] for j in range(GRID)], axis=1) for i in range(GRID)]
        return np.clip(np.concatenate(r, axis=0), 0, 1)

    fig, ax = plt.subplots(rows, 4, figsize=(11, 2.4 * rows))
    for k in range(rows):
        err = np.mean(np.abs(PR[k] - GT[k]), axis=-1)
        ax[k, 0].imshow(tile(X[k])); ax[k, 0].set_xticks([]); ax[k, 0].set_yticks([])
        ax[k, 0].set_ylabel(f"MAE\n{err.mean():.3f}", fontsize=8, rotation=0, labelpad=22, va="center")
        ax[k, 1].imshow(GT[k]); ax[k, 1].axis("off")
        ax[k, 2].imshow(PR[k]); ax[k, 2].axis("off")
        im = ax[k, 3].imshow(err, cmap="magma", vmin=0, vmax=vmax); ax[k, 3].axis("off")
        fig.colorbar(im, ax=ax[k, 3], fraction=0.046, pad=0.04)
    for j, t in enumerate(["Shuffled input", "Ground truth", "Reconstruction", "Abs error"]):
        ax[0, j].set_title(t, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
