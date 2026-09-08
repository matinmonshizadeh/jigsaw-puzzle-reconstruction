#!/usr/bin/env python
"""Train the jigsaw model from scratch.

    python train.py --epochs 60 --batch-size 64 --seed 0 --out runs/exp1

Writes to <out>/: jigsaw.weights.h5 (clean weights of the best epoch), jigsaw_best.weights.h5
(training checkpoint), history.json (per-epoch metrics, best epoch, wall time) and config.yaml
(the configuration actually used). About 2 h on a Colab T4 with the default configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import keras  # noqa: E402
import yaml  # noqa: E402

from jigsaw.data import PatchGenerator, load_stl10, set_seed, split  # noqa: E402
from jigsaw.trainer import build_trainer, count_trainable, make_callbacks  # noqa: E402

PARAM_LIMIT = 6_000_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml", help="YAML with every hyper-parameter")
    p.add_argument("--epochs", type=int, help="override train.epochs (the tau anneal spans the same epochs)")
    p.add_argument("--batch-size", type=int, help="override train.batch_size")
    p.add_argument("--seed", type=int, help="override seed")
    p.add_argument("--out", default="runs/latest", help="output directory")
    p.add_argument("--max-images", type=int, help="use only the first N train/val images (quick debugging runs)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        cfg["sinkhorn"]["anneal_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    os.makedirs(args.out, exist_ok=True)
    set_seed(int(cfg["seed"]))

    images = load_stl10()
    train, val, _ = split(images, int(cfg["data"]["n_train"]), int(cfg["data"]["n_val"]))
    if args.max_images:
        train, val = train[:args.max_images], val[:args.max_images]
    bs = int(cfg["train"]["batch_size"])
    train_gen = PatchGenerator(train, batch_size=bs, shuffle=True)
    val_gen = PatchGenerator(val, batch_size=bs, shuffle=False)

    model = build_trainer(cfg)
    n_params = count_trainable(model)
    print(f"trainable parameters: {n_params:,} (limit {PARAM_LIMIT:,})")
    assert n_params < PARAM_LIMIT
    model.compile(optimizer=keras.optimizers.Adam(float(cfg["train"]["learning_rate"])))

    t0 = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=int(cfg["train"]["epochs"]),
                        callbacks=make_callbacks(cfg, args.out))
    model.load_weights(os.path.join(args.out, "jigsaw_best.weights.h5"))   # best epoch by val_recon

    clean = build_trainer(cfg)                                              # weights only, no optimizer state
    clean.set_weights(model.get_weights())
    clean.save_weights(os.path.join(args.out, "jigsaw.weights.h5"))

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    best = int(min(range(len(hist["val_recon"])), key=hist["val_recon"].__getitem__))
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"history": hist, "best_epoch": best + 1, "best_val_recon": hist["val_recon"][best],
                   "seconds": time.time() - t0, "trainable_params": n_params}, f, indent=1)
    with open(os.path.join(args.out, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"best epoch {best + 1}: val_recon = {hist['val_recon'][best]:.4f} | "
          f"weights -> {os.path.join(args.out, 'jigsaw.weights.h5')}")


if __name__ == "__main__":
    main()
