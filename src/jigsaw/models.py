"""The two sub-networks: OrderNet (patch encoder + self-attention -> 9x9 scores) and RefineNet (U-Net).

Layer order matters for weight files: Keras 3 addresses weights by the position of each layer inside
its model, so these builders must not be reordered if released weights are to stay loadable.
"""
from __future__ import annotations

import keras
from keras import layers

from .data import CROP_SIZE, GRID, IMAGE_SIZE

EMBED_DIM = 128        # token size produced by the patch encoder
N_ATTENTION_BLOCKS = 3
N_HEADS = 4


def build_patch_encoder() -> keras.Model:
    """Shared CNN that embeds one 28x28x3 patch into a 128-d vector."""
    inp = layers.Input((CROP_SIZE, CROP_SIZE, 3))
    x = layers.Conv2D(32, 3, padding="same")(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)   # 14
    x = layers.Conv2D(64, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)   # 7
    x = layers.Conv2D(EMBED_DIM, 3, strides=2, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)  # 4
    x = layers.GlobalAveragePooling2D()(x)                                                                            # 128
    return keras.Model(inp, x, name="patch_encoder")


def build_ordernet() -> keras.Model:
    """9 patches -> (9 slots x 9 cells) score matrix. No positional encoding: the slot order is meaningless."""
    inp = layers.Input((GRID * GRID, CROP_SIZE, CROP_SIZE, 3))
    tokens = layers.TimeDistributed(build_patch_encoder())(inp)          # (9, 128) permutation-equivariant set
    x = tokens
    for _ in range(N_ATTENTION_BLOCKS):                                  # self-attention lets patches compare to each other
        a = layers.LayerNormalization()(x)
        a = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=EMBED_DIM // N_HEADS)(a, a)
        x = layers.Add()([x, a])
        f = layers.LayerNormalization()(x)
        f = layers.Dense(2 * EMBED_DIM, activation="relu")(f)
        f = layers.Dense(EMBED_DIM)(f)
        x = layers.Add()([x, f])
    logits = layers.Dense(GRID * GRID)(x)                                # (9 slots, 9 cells)
    return keras.Model(inp, logits, name="ordernet")


def build_refinenet(base: int = 48) -> keras.Model:
    """Compact U-Net: assembled canvas (+ known-pixel mask) -> full 96x96 RGB image in [0, 1]."""
    def cbr(x, f):
        x = layers.Conv2D(f, 3, padding="same")(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        return x

    inp = layers.Input((IMAGE_SIZE, IMAGE_SIZE, 4))                                     # 3 rgb + 1 known-pixel mask
    c1 = cbr(cbr(inp, base), base);            p1 = layers.MaxPool2D()(c1)               # 96 -> 48
    c2 = cbr(cbr(p1, base * 2), base * 2);     p2 = layers.MaxPool2D()(c2)               # 48 -> 24
    c3 = cbr(cbr(p2, base * 4), base * 4);     p3 = layers.MaxPool2D()(c3)               # 24 -> 12
    b = cbr(cbr(p3, base * 4), base * 4)                                                 # 12
    u3 = layers.UpSampling2D()(b);  u3 = layers.Concatenate()([u3, c3]); u3 = cbr(cbr(u3, base * 4), base * 4)
    u2 = layers.UpSampling2D()(u3); u2 = layers.Concatenate()([u2, c2]); u2 = cbr(cbr(u2, base * 2), base * 2)
    u1 = layers.UpSampling2D()(u2); u1 = layers.Concatenate()([u1, c1]); u1 = cbr(cbr(u1, base), base)
    out = layers.Conv2D(3, 1, activation="sigmoid")(u1)
    return keras.Model(inp, out, name="refinenet")
