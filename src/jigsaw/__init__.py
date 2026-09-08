"""Jigsaw puzzle reconstruction on STL-10: OrderNet + Sinkhorn assembly + RefineNet (Keras 3)."""

__version__ = "1.0.0"

from .data import (PatchGenerator, SeededPatchGenerator, known_pixel_mask, load_stl10,
                   set_seed, split)
from .evaluate import evaluate, mean_patch_baseline, metrics_table, save_error_maps
from .models import build_ordernet, build_patch_encoder, build_refinenet
from .trainer import JigsawTrainer, TauAnneal, build_trainer, count_trainable, make_callbacks

__all__ = [
    "__version__",
    # data
    "PatchGenerator", "SeededPatchGenerator", "known_pixel_mask", "load_stl10", "set_seed", "split",
    # models
    "build_patch_encoder", "build_ordernet", "build_refinenet",
    # trainer
    "JigsawTrainer", "TauAnneal", "build_trainer", "count_trainable", "make_callbacks",
    # evaluation
    "evaluate", "mean_patch_baseline", "metrics_table", "save_error_maps",
]
