import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback


class CyclicLRCallback(Callback):
    """
    Saugat 11/14/21:
    Adapted (with slight modifications) from:
    https://github.com/bckenstler/CLR/blob/master/clr_callback.py

    Saugat 11/15/21:
    The major change is the adaptation for the LossScaleOptimizer.


    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
        use_loss_scale_optimizer: bool
            Specify whether the LossScaleOptimizer is being used to wrap
            the original optimizer.
        use_tensorboard: bool
            Specify whether to add tensorboard summaries.
    """

    def __init__(
        self,
        base_lr=0.001,
        max_lr=0.006,
        step_size=2000.0,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        use_loss_scale_optimizer=False,
        use_tensorboard=False,
    ):
        super(CyclicLRCallback, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.use_loss_scale_optimizer = use_loss_scale_optimizer
        self.use_tensorboard = use_tensorboard
        self.clr_iterations = 0
        self.trn_iterations = 0
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.0

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == "cycle":
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations
            )

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.use_loss_scale_optimizer:
            optimizer = self.model.optimizer._optimizer
        else:
            optimizer = self.model.optimizer

        if self.clr_iterations == 0:
            lr_this = self.base_lr
        else:
            lr_this = self.clr()

        K.set_value(optimizer.lr, lr_this)
        logs.update({"lr": lr_this})
        super().on_train_begin(logs)

    def on_batch_end(self, iteration, logs=None):
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        if self.use_loss_scale_optimizer:
            optimizer = self.model.optimizer._optimizer
        else:
            optimizer = self.model.optimizer

        # self.history.setdefault('lr', []).append(K.get_value(optimizer.lr))
        # self.history.setdefault('iterations', []).append(self.trn_iterations)

        lr_this = self.clr()
        K.set_value(optimizer.lr, lr_this)

        if self.use_tensorboard:
            tf.summary.scalar("lr", data=lr_this, step=int(self.trn_iterations))
        logs.update({"lr": lr_this})
        super().on_batch_end(iteration, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({"lr_per_epoch": self.clr()})
        super().on_epoch_end(epoch, logs)


class HistoryWriterCallback(Callback):
    """Callback to write the history to a file. Append if history already exists."""

    def __init__(self, hist_train, save_fname):
        super(HistoryWriterCallback, self).__init__()
        self.hist_train = hist_train
        self.save_fname = save_fname
        self._epoch_prev = 0

    def set_params(self, params):
        for k, val in params.items():
            self.hist_train.params.setdefault(k, [])
            self.hist_train.params[k].append(val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch == 0:
            if len(self.hist_train.epochs_per_fit) > 0:
                self._epoch_prev = self.hist_train.epochs_per_fit[-1][-1] + 1
            self.hist_train.epochs_per_fit.append([0])
        else:
            self.hist_train.epochs_per_fit[-1].append(epoch)

        if len(self.hist_train.epoch) == 0:
            self.hist_train.epoch = np.array([0])
        else:
            self.hist_train.epoch = np.append(self.hist_train.epoch, epoch + self._epoch_prev)

        for k, v in logs.items():
            self.hist_train.history.setdefault(k, []).append(v)

        self.hist_train.save(self.save_fname)
