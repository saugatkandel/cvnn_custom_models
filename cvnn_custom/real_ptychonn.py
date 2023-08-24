import joblib
import numpy as np
import tensorflow as tf

import cvnn_custom.callbacks as callbacks
import cvnn_custom.utils as cutils
from cvnn_custom.graph_utils import get_dot_graph

from . import real_models as rmods
from .params import ConvParams, DownConvParams, UpConvParams
from .trainer import Trainer

DEFAULT_REAL_ACTIVATION = "leaky_relu"


def complex_loss_metric(y_true, y_pred):
    y_cast = tf.saturate_cast(y_true, "float16")
    mag = y_cast[0][..., None]
    ph = y_cast[1][..., None]
    mag_pred, ph_pred = y_pred

    r1, i1 = mag * tf.math.cos(ph), mag * tf.math.sin(ph)
    r2, i2 = mag_pred * tf.math.cos(ph_pred), mag_pred * tf.math.sin(ph_pred)

    norm = tf.norm([r1 - r2, i1 - i2], ord=1, axis=-1)

    return tf.math.reduce_mean(norm, axis=-1)


class CustomRealPtychoNNModel(tf.keras.Model):
    def compute_metrics(self, x, y, y_pred, sample_weight):
        metrics_results = super().compute_metrics(x, y, y_pred, sample_weight)
        metrics_results["cmplx_loss"] = complex_loss_metric(y, y_pred)
        return metrics_results


def create_model_64_to_128_strides(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    real_activation: str = DEFAULT_REAL_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="float32")
    down_stack = [input_img]

    for ix, nf in enumerate([32, 64, 128]):
        down_layer = None if ix == 0 else 0
        down_stack += [
            *rmods.block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_down_stride_in_layer=down_layer,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    mag_stack = rmods.block_fn(
        128,
        down_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=real_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn_mag",
    )
    ph_stack = rmods.block_fn(
        128,
        down_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=real_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn_ph",
    )

    for ix, nf in enumerate([64, 32, 16]):
        mag_stack += [
            *rmods.block_fn(
                nf,
                mag_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"mag_ul_{ix}",
                n_layers=2,
            )
        ]
        ph_stack += [
            *rmods.block_fn(
                nf,
                ph_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"ph_ul_{ix}",
                n_layers=2,
            )
        ]

    mag_out_layer = rmods.conv_fn(1, conv_params=None, name=f"mag_ul_{ix}", activation="linear")(mag_stack[-1])
    ph_out_layer = rmods.conv_fn(1, conv_params=None, name=f"ph_ul_{ix}", activation="linear")(ph_stack[-1])

    mag_out = tf.keras.layers.Identity(name="mag")(tf.keras.activations.sigmoid(mag_out_layer))
    ph_out = tf.keras.layers.Identity(name="ph")(tf.keras.activations.tanh(ph_out_layer) * np.pi)

    model = CustomRealPtychoNNModel(inputs=[input_img], outputs=[mag_out, ph_out])
    return model


def create_model_64_to_128_upsamp(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    real_activation: str = DEFAULT_REAL_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="float32")
    down_stack = [input_img]

    for ix, nf in enumerate([32, 64, 128]):
        down_stack += [
            *rmods.block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_pool_after_layer=1,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    mag_stack = rmods.block_fn(
        128,
        down_stack[-1],
        use_upsamp_after_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=real_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn_mag",
    )
    ph_stack = rmods.block_fn(
        128,
        down_stack[-1],
        use_upsamp_after_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=real_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn_ph",
    )

    for ix, nf in enumerate([64, 32, 16]):
        mag_stack += [
            *rmods.block_fn(
                nf,
                mag_stack[-1],
                use_upsamp_after_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"mag_ul_{ix}",
                n_layers=2,
            )
        ]
        ph_stack += [
            *rmods.block_fn(
                nf,
                ph_stack[-1],
                use_upsamp_after_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=real_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"ph_ul_{ix}",
                n_layers=2,
            )
        ]

    mag_out_layer = rmods.conv_fn(1, conv_params=None, name=f"mag_ul_{ix}", activation="sigmoid")(mag_stack[-1])
    ph_out_layer = rmods.conv_fn(1, conv_params=None, name=f"ph_ul_{ix}", activation="tanh")(ph_stack[-1])

    mag_out = tf.keras.layers.Identity(name="mag")(mag_out_layer)
    ph_out = tf.keras.layers.Identity(name="ph")(ph_out_layer * np.pi)

    model = CustomRealPtychoNNModel(inputs=[input_img], outputs=[mag_out, ph_out])
    return model


class RealTrainer(Trainer):
    def setTrainingData(self, X_train_full: np.ndarray, Y_train_full: np.ndarray, valid_data_ratio: float = 0.1):
        self.H, self.W = X_train_full.shape[-2:]

        self.X_train_full = X_train_full
        self.Y_train_full = Y_train_full
        self.ntrain_full = self.X_train_full.shape[0]

        self.valid_data_ratio = valid_data_ratio
        self.nvalid = int(self.ntrain_full * self.valid_data_ratio)
        self.ntrain = self.ntrain_full - self.nvalid

        self.Y_train_mag = np.abs(self.Y_train_full)
        self.Y_train_ph = np.angle(self.Y_train_full)

        self.x_dataset = tf.data.Dataset.from_tensor_slices(X_train_full)
        self.y_mag_dataset = tf.data.Dataset.from_tensor_slices(self.Y_train_mag)
        self.y_ph_dataset = tf.data.Dataset.from_tensor_slices(self.Y_train_ph)

        self.y_dataset = tf.data.Dataset.zip((self.y_mag_dataset, self.y_ph_dataset))

        self.ds_train_full = tf.data.Dataset.zip((self.x_dataset, self.y_dataset))
        self.ds_train_full = self.ds_train_full.shuffle(self.ntrain_full, seed=1, reshuffle_each_iteration=False)

        self.ds_validation = self.ds_train_full.take(self.nvalid)
        self.ds_train = self.ds_train_full.skip(self.nvalid)

        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle(self.ntrain, reshuffle_each_iteration=True)
        self.ds_train = self.ds_train.batch(self.batch_size, drop_remainder=True)
        self.ds_train = self.ds_train.prefetch(tf.data.AUTOTUNE)

        self.ds_validation = self.ds_validation.cache()
        self.ds_validation = self.ds_validation.batch(self.batch_size)
        self.ds_validation = self.ds_validation.prefetch(tf.data.AUTOTUNE)

        self.iters_per_epoch = int(np.floor(self.ntrain / self.batch_size))

        # Final batch will be less than batch size

    def setTestData(self, X_test: np.ndarray, Y_test: np.ndarray):
        self.H, self.W = X_test.shape[-2:]

        self.X_test = X_test
        self.Y_test = Y_test
        self.ntest = self.X_test.shape[0]

        ds_test = tf.data.Dataset.from_tensor_slices(X_test)
        ds_test = ds_test.batch(self.batch_size, drop_remainder=False)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
        self.ds_test = ds_test

    """
    def setOptimizationParams(
        self,
        epochs_per_half_cycle: int = 6,
        max_lr: float = 5e-4,
        min_lr: float = 1e-4,
        use_loss_scale_optimizer: bool = True,
    ):
        # Optimizer details

        self.epochs_per_half_cycle = epochs_per_half_cycle
        self.iters_per_half_cycle = (
            epochs_per_half_cycle * self.iters_per_epoch
        )  # Paper recommends 2-10 number of iterations
        self.use_loss_scale_optimizer = use_loss_scale_optimizer

        print(
            f"LR step size is: {self.iters_per_half_cycle}, "
            f"which is every {self.iters_per_half_cycle / self.iters_per_epoch} epochs."
        )

        self.max_lr = max_lr
        self.min_lr = min_lr

        # adabelief = tfa.optimizers.AdaBelief(1e-3)
        # ranger = tfa.optimizers.Lookahead(adabelief, sync_period=6, slow_step_size=0.5)

        # self.

        self.optimizer = tf.keras.optimizers.Adam(max_lr)

        if use_loss_scale_optimizer:
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        self.model.compile(
            optimizer=self.optimizer,
            loss="mean_absolute_error",
            # run_eagerly=True,
            # jit_compile=True,
            #    # steps_per_execution=self.iters_per_epoch,
        )
    

    def initModel(
        self,
        model_params_path: str = None,
        history_path: str = None,
        verbose=True,
        gen_dot_graph: bool = False,
        graph_save_name: str = None,
    ):
        self.load_model_params_path = model_params_path
        self.load_history_path = history_path
        if model_params_path is not None:
            self.model.load_weights(model_params_path)

        if history_path is not None:
            self.history = cutils.HistoryTrainedModel.load(history_path)
        else:
            self.history = cutils.HistoryTrainedModel()

        if verbose:
            print(cutils.full_summary(self.model))
            if graph_save_name is not None:
                graph_save_name = f"{self.output_path}/{graph_save_name}"
            get_dot_graph(self.model, file_save_name=graph_save_name)

        # self.lr_log_cb = callbacks.LearningRateLogCallback()

    def initTrainingRun(
        self,
        early_stop_patience: int = 10,
        early_stop_loss: str = "val_loss",
        early_stop_delta: float = 0,
        model_checkpoint_losses: list = ["val_loss", "loss"],
        # overwrite_model: bool = True,
        overwrite_history: bool = True,
        best_loss_checkpoint_frequency: int = 1,
        extra_checkpoint_frequency: int = 0,
        use_tensorboard: bool = False,
    ):
        # self.overwrite_model = overwrite_model
        self.overwrite_history = overwrite_history
        self.use_tensorboard = use_tensorboard
        self.model_checkpoint_losses = model_checkpoint_losses
        self.best_loss_checkpoint_frequency = best_loss_checkpoint_frequency
        self.extra_checkpoint_frequency = extra_checkpoint_frequency

        datetime_str = datetime.now().strftime("-%Y%m%d-%H%M%S")

        self.history_save_fname = self.output_path + "/histories" + self.output_suffix + datetime_str
        if self.load_history_path is not None and self.overwrite_history:
            self.history_save_fname = self.load_history_path

        self.model_save_fnames = []
        self.checkpoint_cbs = []
        for loss in model_checkpoint_losses:
            fname = self.output_path + f"/{loss}_best_model" + self.output_suffix + datetime_str + ".h5"
            cb = tf.keras.callbacks.ModelCheckpoint(
                fname,
                monitor=loss,
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode="auto",
                save_freq=self.iters_per_epoch * self.best_loss_checkpoint_frequency,
            )
            self.model_save_fnames.append(fname)
            self.checkpoint_cbs.append(cb)

        if self.extra_checkpoint_frequency > 0:
            fname = self.output_path + "/model" + self.output_suffix + "_epoch_{epoch}.h5"
            cb = tf.keras.callbacks.ModelCheckpoint(
                fname,
                monitor="loss",
                verbose=1,
                save_best_only=False,
                save_weights_only=True,
                mode="auto",
                save_freq=self.iters_per_epoch * self.extra_checkpoint_frequency,
            )
            # save_freq=self.extra_checkpoint_frequency * self.iters_per_epoch)
            self.checkpoint_cbs.append(cb)

        print("output path is", self.output_path)
        print(os.path.isdir(self.output_path))
        if not os.path.isdir(self.output_path):
            print("creating output dir", self.output_path)
            os.mkdir(self.output_path)

        if self.use_tensorboard:
            logdir = self.output_path + "/logs/scalars/" + datetime_str
            tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            file_writer = tf.summary.create_file_writer(logdir + "/metrics")
            file_writer.set_as_default()

        # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(self.model_save_fname,
        #                                        monitor=monitor_loss, verbose=1, save_best_only=True,
        #                                        save_weights_only=True, mode='auto', save_freq='epoch')
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            early_stop_loss, patience=early_stop_patience, min_delta=early_stop_delta
        )
        hist_writer_cb = callbacks.HistoryWriterCallback(self.history, self.history_save_fname)
        lr_cb = callbacks.CyclicLRCallback(
            base_lr=self.min_lr,
            max_lr=self.max_lr,
            mode="triangular2",
            step_size=self.iters_per_half_cycle,
            use_loss_scale_optimizer=self.use_loss_scale_optimizer,
            use_tensorboard=self.use_tensorboard,
        )
        # reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.2, patience=5, min_lr=self.min_lr)
        # self.callbacks = [checkpoint_cb, early_stop_cb, cyclic_lr_cb, hist_writer_cb]
        self.callbacks = [*self.checkpoint_cbs, early_stop_cb, hist_writer_cb, lr_cb]
        if self.use_tensorboard:
            self.callbacks.append(tensorboard_cb)

        self.is_training_initialized = True

    def run(self, epochs: int, output_frequency: int = 1, initial_epoch=0, run_verbosity=1):
        if not self.is_training_initialized:
            raise ValueError("First call the initTrainingRun to set up the training run before calling run.")
        return self.model.fit(
            self.ds_train,
            validation_data=self.ds_validation,
            epochs=epochs,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch,
            verbose=run_verbosity,
        )
    """

    def predictTestData(self, joblib_save_path: str = None):
        # self.model.compile(jit_compile=True)

        self.Y_mag_eval, self.Y_ph_eval = self.model.predict(self.ds_test)
        self.Y_mag_eval = np.squeeze(self.Y_mag_eval)
        self.Y_ph_eval = np.squeeze(self.Y_ph_eval)

        self.Y_eval = self.Y_mag_eval * np.exp(1j * self.Y_ph_eval)
        dat = {"mag": self.Y_mag_eval, "ph": self.Y_ph_eval}

        print("Saving inferences...")
        if joblib_save_path is not None:
            joblib.dump(dat, joblib_save_path, compress=("lz4", 3))
        print("Saved!")

        return self.Y_eval
