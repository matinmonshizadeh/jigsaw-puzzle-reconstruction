"""JigsawTrainer: OrderNet -> Sinkhorn -> differentiable assembly -> RefineNet, trained on (X, Y) only.

Losses and metrics
- ``pmae`` (placement MAE): MAE between the assembled canvas and the target on the known pixels only.
  Every patch is an exact crop of the target, so this is zero only when every patch sits in its own
  cell: label-free routing supervision.
- ``recon`` (reconstruction MAE): MAE between the final image and the target over all pixels.
- ``place``: training-time routing monitor recovered from Y by nearest content (a lower bound; the
  exact metric, computed from the true permutation, lives in :mod:`jigsaw.evaluate`).
- ``tau``: the Sinkhorn temperature actually used inside the compiled training graph.

The attribute names of this class (``ordernet``, ``refinenet``, the four metrics) define the layout
of the ``.weights.h5`` files; keep them if released weights are to stay loadable.
"""
from __future__ import annotations

import os

import keras
import numpy as np
import tensorflow as tf

from .data import CROP_SIZE, GRID, IMAGE_SIZE, MARGIN, PATCH_SIZE, known_pixel_mask
from .models import build_ordernet, build_refinenet


class JigsawTrainer(keras.Model):
    def __init__(self, ordernet, refinenet, w_place=1.0, train_tau=1.0, eval_tau=0.01, n_iters=20, **kw):
        super().__init__(**kw)
        self.ordernet, self.refinenet = ordernet, refinenet
        self.w_place, self.eval_tau, self.n_iters = w_place, eval_tau, n_iters
        # Non-trainable tf.Variable: the annealing callback updates it while train_step runs as a
        # compiled graph. It is not a model weight (not counted in the budget, not saved).
        self.train_tau = tf.Variable(train_tau, trainable=False, dtype=tf.float32, name="train_tau")
        self.mask = tf.constant(known_pixel_mask().reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1))
        self.loss_t = keras.metrics.Mean(name="loss")
        self.recon_t = keras.metrics.Mean(name="recon")
        self.pmae_t = keras.metrics.Mean(name="pmae")
        self.place_t = keras.metrics.Mean(name="place")

    @property
    def metrics(self):
        return [self.loss_t, self.recon_t, self.pmae_t, self.place_t]

    # ---- differentiable pieces -------------------------------------------------------------------
    def sinkhorn(self, A, tau):
        """Log-domain Sinkhorn: alternating row / column log-softmax normalisations (no parameters).

        The last step normalises the columns, so every destination cell receives a convex combination
        of the patches. Where OrderNet is confident the result is a permutation for all practical
        purposes; for exact ties it is a soft blend of the tied patches.
        """
        la = A / tau
        for _ in range(self.n_iters):
            la = la - tf.reduce_logsumexp(la, axis=2, keepdims=True)   # rows    (slot i) sum to 1
            la = la - tf.reduce_logsumexp(la, axis=1, keepdims=True)   # columns (cell j) sum to 1
        return tf.exp(la)

    def assemble(self, X, P):
        """Place the patches on a 96x96 canvas: cell j <- sum_i P[i, j] * patch_i (zeros on the seams)."""
        pos = tf.einsum("bij,bihwc->bjhwc", P, X)
        pos = tf.pad(pos, [[0, 0], [0, 0], [MARGIN, MARGIN], [MARGIN, MARGIN], [0, 0]])   # centre in a 32x32 cell
        pos = tf.reshape(pos, (-1, GRID, GRID, PATCH_SIZE, PATCH_SIZE, 3))              # (B, r, c, h, w, ch)
        pos = tf.transpose(pos, [0, 1, 3, 2, 4, 5])                                      # (B, r, h, c, w, ch)
        return tf.reshape(pos, (-1, IMAGE_SIZE, IMAGE_SIZE, 3))

    def extract_cells(self, Y):
        """Inverse of the assembly on the known pixels: the 9 centre crops of Y in grid order."""
        y = tf.reshape(Y, (-1, GRID, PATCH_SIZE, GRID, PATCH_SIZE, 3))                  # (B, r, h, c, w, ch)
        y = tf.transpose(y, [0, 1, 3, 2, 4, 5])                                          # (B, r, c, h, w, ch)
        y = tf.reshape(y, (-1, GRID * GRID, PATCH_SIZE, PATCH_SIZE, 3))
        return y[:, :, MARGIN:MARGIN + CROP_SIZE, MARGIN:MARGIN + CROP_SIZE, :]

    def forward(self, X, tau, training=False):
        """Returns (P, canvas, final) for a batch of shuffled patches."""
        P = self.sinkhorn(self.ordernet(X, training=training), tau)
        C = self.assemble(X, P)
        Mb = tf.broadcast_to(self.mask, (tf.shape(C)[0], IMAGE_SIZE, IMAGE_SIZE, 1))
        ref = self.refinenet(tf.concat([C, Mb], axis=-1), training=training)
        final = self.mask * C + (1.0 - self.mask) * ref
        return P, C, final

    def call(self, X, training=False):
        tau = self.train_tau if training else self.eval_tau
        return self.forward(X, tau, training)[2]

    # ---- losses and monitors ---------------------------------------------------------------------
    def masked_mae(self, C, Y):
        num = tf.reduce_sum(tf.abs(C - Y) * self.mask)
        den = tf.reduce_sum(self.mask) * tf.cast(tf.shape(C)[0], tf.float32) * 3.0
        return num / den

    def routing_accuracy(self, X, Y, P):
        """Training-time monitor only: the 'true' cell of each patch is recovered from Y by nearest content.
        Exact whenever the 9 patches are distinct; pixel-identical patches make it a lower bound."""
        cells = self.extract_cells(Y)
        Xf = tf.reshape(X, (tf.shape(X)[0], GRID * GRID, 1, -1))
        Cf = tf.reshape(cells, (tf.shape(X)[0], 1, GRID * GRID, -1))
        D = tf.reduce_sum(tf.abs(Xf - Cf), axis=-1)
        true_pos = tf.argmin(D, axis=2, output_type=tf.int32)
        pred_pos = tf.argmax(P, axis=2, output_type=tf.int32)
        return tf.reduce_mean(tf.cast(true_pos == pred_pos, tf.float32))

    def _jigsaw_losses(self, X, Y, tau, training):
        P, C, final = self.forward(X, tau, training)
        pmae = self.masked_mae(C, Y)
        recon = tf.reduce_mean(tf.abs(final - Y))
        return P, pmae, recon, self.w_place * pmae + recon

    def train_step(self, data):
        X, Y = data
        with tf.GradientTape() as tape:
            P, pmae, recon, loss = self._jigsaw_losses(X, Y, self.train_tau, True)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables),
                                            self.trainable_variables))
        acc = self.routing_accuracy(X, Y, P)
        self.loss_t.update_state(loss); self.recon_t.update_state(recon)
        self.pmae_t.update_state(pmae); self.place_t.update_state(acc)
        logs = {m.name: m.result() for m in self.metrics}
        logs["tau"] = tf.identity(self.train_tau)          # the temperature actually used inside the graph
        return logs

    def test_step(self, data):
        X, Y = data
        P, pmae, recon, loss = self._jigsaw_losses(X, Y, self.eval_tau, False)
        acc = self.routing_accuracy(X, Y, P)
        self.loss_t.update_state(loss); self.recon_t.update_state(recon)
        self.pmae_t.update_state(pmae); self.place_t.update_state(acc)
        return {m.name: m.result() for m in self.metrics}


class TauAnneal(keras.callbacks.Callback):
    """Linearly anneal ``train_tau`` from ``tau_start`` to ``tau_end`` over ``anneal_epochs``, then hold."""

    def __init__(self, tau_start=1.0, tau_end=0.2, anneal_epochs=60):
        super().__init__()
        self.tau_start, self.tau_end, self.anneal_epochs = tau_start, tau_end, anneal_epochs

    def on_epoch_begin(self, epoch, logs=None):
        frac = min(1.0, epoch / max(1, self.anneal_epochs - 1))          # 0 -> 1, then held
        self.model.train_tau.assign(self.tau_start + (self.tau_end - self.tau_start) * frac)


def count_trainable(model: keras.Model) -> int:
    """Number of trainable scalars (BatchNorm moving statistics are non-trainable and excluded)."""
    return int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))


def build_trainer(cfg: dict) -> JigsawTrainer:
    """Build OrderNet + RefineNet + trainer from a config dict (``configs/default.yaml``) and create its variables."""
    s, l, m = cfg["sinkhorn"], cfg["loss"], cfg["model"]
    model = JigsawTrainer(build_ordernet(), build_refinenet(base=int(m["base"])),
                          w_place=float(l["w_place"]), train_tau=float(s["tau_start"]),
                          eval_tau=float(s["eval_tau"]), n_iters=int(s["n_iters"]))
    model(tf.zeros((2, GRID * GRID, CROP_SIZE, CROP_SIZE, 3)))          # build variables
    return model


def make_callbacks(cfg: dict, out_dir: str) -> list:
    """Temperature annealing, LR halving on plateau, early stopping and best-checkpoint saving."""
    t, s = cfg["train"], cfg["sinkhorn"]
    monitor = t.get("monitor", "val_recon")
    return [
        TauAnneal(float(s["tau_start"]), float(s["tau_end"]), int(s["anneal_epochs"])),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=float(t["reduce_lr"]["factor"]),
                                          patience=int(t["reduce_lr"]["patience"]), mode="min", verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="min", patience=int(t["early_stopping"]["patience"]),
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "jigsaw_best.weights.h5"), monitor=monitor,
                                        mode="min", save_best_only=True, save_weights_only=True, verbose=1),
    ]
