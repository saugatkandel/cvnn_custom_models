from .trainer import Trainer
import tensorflow as tf


class ProbeTrainer(Trainer):
    def brightfield_loss_fn(self, y_true, y_pred):
        def clip_fn(y):
            amps = tf.abs(y)
            amax2 = tf.reduce_max(amps, axis=0, keepdims=True)
            amps2 = tf.where(amps < amax2 / 1e2, tf.zeros_like(amps), amps)
            y2 = y * tf.complex(amps2 / amps, 0.0)
            return y2

        base_loss_fn = tf.keras.losses.mean_absolute_error
        y_true_thres = clip_fn(y_true)
        y_pred_thres = clip_fn(y_pred)
        return tf.keras.losses.cosine_similarity(y_true, y_pred, axis=(1, 2))

    def _compile_custom(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.brightfield_loss_fn,
            jit_compile=True,
            #    # steps_per_execution=self.iters_per_epoch,
        )
