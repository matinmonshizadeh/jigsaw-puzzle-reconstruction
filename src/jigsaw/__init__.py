"""Jigsaw puzzle reconstruction on STL-10: OrderNet + Sinkhorn assembly + RefineNet (Keras 3)."""

__version__ = "1.0.0"

from .data import (PatchGenerator, SeededPatchGenerator, known_pixel_mask, load_stl10,  # noqa: F401
                   set_seed, split)
from .evaluate import evaluate, mean_patch_baseline, metrics_table, save_error_maps  # noqa: F401
from .models import build_ordernet, build_patch_encoder, build_refinenet  # noqa: F401
from .trainer import JigsawTrainer, TauAnneal, build_trainer, count_trainable, make_callbacks  # noqa: F401
