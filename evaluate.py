#!/usr/bin/env python
"""Evaluate trained weights on the seeded test split and save the error-map figure.

    python evaluate.py --weights weights/jigsaw.weights.h5

Prints the metrics table (mean-patch baseline vs. ours) and writes docs/error_maps.png.
About 1 minute on a Colab T4 after the STL-10 download, about 8 minutes on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import yaml  # noqa: E402

from jigsaw.data import load_stl10, set_seed, split  # noqa: E402
from jigsaw.evaluate import evaluate, metrics_table, save_error_maps  # noqa: E402
from jigsaw.trainer import build_trainer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="path to jigsaw.weights.h5")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0, help="seed of the evaluation permutations")
    p.add_argument("--error-maps", default="docs/error_maps.png", help="output figure ('' to skip)")
    p.add_argument("--rows", type=int, default=6, help="samples shown in the error-map figure")
    p.add_argument("--json", default="", help="optional path to write the metrics as JSON")
    p.add_argument("--max-images", type=int, help="evaluate only the first N test images (debugging)")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)

    images = load_stl10()
    _, _, test = split(images, int(cfg["data"]["n_train"]), int(cfg["data"]["n_val"]))

    model = build_trainer(cfg)
    model.load_weights(args.weights)

    res = evaluate(model, test, batch_size=args.batch_size, seed=args.seed, max_images=args.max_images)
    print(metrics_table(res))
    if args.error_maps:
        os.makedirs(os.path.dirname(args.error_maps) or ".", exist_ok=True)
        save_error_maps(model, test, args.error_maps, rows=args.rows, seed=args.seed + 1)
        print(f"error maps -> {args.error_maps}")
    if args.json:
        scalars = {k: v for k, v in res.items() if not k.endswith("per_image_mae")}
        with open(args.json, "w") as f:
            json.dump(scalars, f, indent=1)
        print(f"metrics -> {args.json}")
    return res


if __name__ == "__main__":
    main()
