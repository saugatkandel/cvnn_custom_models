import dataclasses as dt
from typing import Tuple

import tensorflow as tf
from cvnn.activations import cart_leaky_relu, crelu, linear, modrelu, zrelu
from cvnn.layers import (
    ComplexBatchNormalization,
    ComplexConv2D,
    ComplexConv2DTranspose,
    ComplexMaxPooling2D,
    ComplexUpSampling2D,
)
from keras.engine import data_adapter

from .params import ConvParams, DownConvParams, UpConvParams

DEFAULT_ACTIVATION = "cleakyrelu"


class CustomModelForJitCorrection(tf.keras.Model):
    def train_step(self, data):
        """The logic for one training step.
        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.
        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.
        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """

        @tf.function(jit_compile=True, reduce_retracing=True)
        def _train_step_internal():
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(x, y, y_pred, sample_weight)
            self._validate_target_and_loss(y, loss)

            grads = tape.gradient(loss, self.trainable_variables)
            return y_pred, grads

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred, grads = _train_step_internal()
        grads_and_vars = list(zip(grads, self.trainable_variables))
        self.optimizer.apply_gradients(grads_and_vars)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def test_step(self, data):
        """The logic for one evaluation step.
        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.
        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.
        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.
        Args:
          data: A nested structure of `Tensor`s.
        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned.
        """

        @tf.function(jit_compile=True, reduce_retracing=True)
        def _test_step_internal():
            y_pred = self(x, training=False)
            # Updates stateful loss metrics.
            self.compute_loss(x, y, y_pred, sample_weight)
            return y_pred

        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = _test_step_internal()
        return self.compute_metrics(x, y, y_pred, sample_weight)


def get_activation(activation: str):
    activations = {
        "crelu": crelu,
        "zrelu": zrelu,
        "modrelu": modrelu,
        "cleakyrelu": cart_leaky_relu,
        "linear": linear,
    }
    if activation is not None:
        activation = activations[activation]
    return activation


def conv_fn(
    n_filters: int,
    conv_params: ConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    name: str = None,
):
    if conv_params is None:
        conv_params = ConvParams()

    return ComplexConv2D(
        n_filters,
        activation=get_activation(activation),
        **dt.asdict(conv_params),
        name=name,
    )


def up_conv_fn(
    n_filters: int,
    up_conv_params: UpConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    name: str = None,
):
    if up_conv_params is None:
        up_conv_params = UpConvParams()

    return ComplexConv2DTranspose(
        n_filters,
        activation=get_activation(activation),
        **dt.asdict(up_conv_params),
        name=name,
    )


def down_conv_fn(
    n_filters: int,
    down_conv_params: DownConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    name: str = None,
):
    if down_conv_params is None:
        down_conv_params = DownConvParams()

    return ComplexConv2D(
        n_filters,
        activation=get_activation(activation),
        **dt.asdict(down_conv_params),
        name=name,
    )


def pool_fn(
    pool_size: Tuple = (2, 2),
    padding: str = "same",
    data_format: str = "channels_last",
    **kwargs,
):
    return ComplexMaxPooling2D(pool_size=pool_size, padding=padding, data_format=data_format, **kwargs)


def upsample_fn(
    size: Tuple = (2, 2),
    interpolation: str = "bilinear",
    data_format: str = "channels_last",
    **kwargs,
):
    return ComplexUpSampling2D(size=size, interpolation=interpolation, data_format=data_format, **kwargs)


def block_fn(
    n_filters: int,
    l0: tf.keras.layers.Layer,
    use_down_stride_in_layer: int = None,
    use_up_stride_in_layer: int = None,
    use_pool_after_layer: int = None,
    use_upsamp_after_layer: int = None,
    apply_batchnorm: bool = False,
    n_layers: int = 2,
    activation: str = DEFAULT_ACTIVATION,
    conv_params: ConvParams = None,
    up_conv_params: ConvParams = None,
    down_conv_params: ConvParams = None,
    name=None,
):
    if (use_down_stride_in_layer is not None) and (use_up_stride_in_layer is not None):
        if use_down_stride_in_layer == use_up_stride_in_layer:
            raise ValueError
    activation = get_activation(activation)
    block = []
    for ix in range(0, n_layers):
        if use_down_stride_in_layer == ix:
            block += [
                down_conv_fn(
                    n_filters,
                    down_conv_params=down_conv_params,
                    activation=None,
                    name=f"{name}_dconv{ix}",
                )(l0)
            ]
        elif use_up_stride_in_layer == ix:
            block += [
                up_conv_fn(
                    n_filters,
                    up_conv_params=up_conv_params,
                    activation=None,
                    name=f"{name}_uconv{ix}",
                )(l0)
            ]
        else:
            block += [
                conv_fn(
                    n_filters,
                    conv_params=conv_params,
                    activation=None,
                    name=f"{name}_conv{ix}",
                )(l0)
            ]

        if apply_batchnorm and (ix == n_layers - 1):
            block += [ComplexBatchNormalization(epsilon=1e-5)(block[-1])]

        block += [activation(block[-1])]
        l0 = block[-1]
        l0 = activation(l0)

        if use_pool_after_layer == ix:
            block += [pool_fn(name=f"{name}_pool")(l0)]
            l0 = block[-1]
        if use_upsamp_after_layer == ix:
            block += [upsample_fn(name=f"{name}_upsamp")(l0)]
            l0 = block[-1]

    return block


def create_basic_model(img_h, img_w):
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")
    l1 = conv_fn(32)

    down_stack = [
        conv_fn(32),
        down_conv_fn(32),
        conv_fn(64),
        down_conv_fn(64),
        conv_fn(128),
        down_conv_fn(128),
    ]
    up_stack = [
        conv_fn(128),
        up_conv_fn(128),
        conv_fn(64),
        up_conv_fn(64),
        conv_fn(32),
        up_conv_fn(32),
    ]
    last = ComplexConv2D(1, (3, 3), padding="same", data_format="channels_last")

    x = input_img
    for layer in down_stack:
        x = layer(x)

    for layer in up_stack:
        x = layer(x)

    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model


def create_unet_model(img_h, img_w):
    # This Unet implementation is different from that in the cvnn examples directory. I need to check this carefully.
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")

    down_stack = [
        conv_fn(32),
        down_conv_fn(32),
        conv_fn(64),
        down_conv_fn(64),
        conv_fn(128),
        down_conv_fn(128),
    ]
    up_stack = [
        conv_fn(128),
        up_conv_fn(128),
        conv_fn(64),
        up_conv_fn(64),
        conv_fn(32),
        up_conv_fn(32),
    ]

    last = ComplexConv2D(1, (3, 3), padding="same", data_format="channels_last")
    concat = tf.keras.layers.Concatenate()

    x = input_img
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips)

    # Upsampling and establishing the skip connections
    for indx, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        if (indx + 1) % 2 == 0:
            print(x, skip)
            x = concat([x, skip])

    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model


def create_resnet_model(img_h, img_w):
    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")

    down_stack = [
        conv_fn(32),
        down_conv_fn(32),
        conv_fn(64),
        down_conv_fn(64),
        conv_fn(128),
        down_conv_fn(128),
    ]
    up_stack = [
        conv_fn(128),
        up_conv_fn(128),
        conv_fn(64),
        up_conv_fn(64),
        conv_fn(32),
        up_conv_fn(32),
    ]

    last = ComplexConv2D(1, (3, 3), padding="same", data_format="channels_last")
    concat = tf.keras.layers.Concatenate()

    x = input_img
    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips)

    # Upsampling and establishing the skip connections
    for indx, (up, skip) in enumerate(zip(up_stack, skips)):
        x = up(x)
        x = x - skip
        # if (indx + 1) % 2 == 0:
        #    print(x, skip)
        #    x = concat([x, skip])

    decoded = last(x)
    model = tf.keras.Model(input_img, decoded)
    return model


class AmplitudeThreshold(tf.keras.layers.Layer):
    """Apply a threshold to the amplitude in a complex-valued NN.
    Implementing this as a layer to make the model summary cleaner.
    """

    def __init__(self, a_min=0.0, a_max=1.0, **kwargs):
        super(AmplitudeThreshold, self).__init__(**kwargs)
        if a_min < 0 and a_max <= a_min:
            raise ValueError("The supplied thresholds have to be 0 <= a_min < a_max.")
        self.a_min = a_min
        self.a_max = a_max

    def call(self, inputs):
        amps = tf.math.abs(inputs)
        amps_clipped = tf.clip_by_value(amps, self.a_min, self.a_max)
        outputs = inputs * tf.cast(amps_clipped / (amps + 1e-8), tf.complex64)
        return outputs

    def get_config(self):
        config = {"a_min": float(self.a_min), "a_max": float(self.a_max)}
        base_config = super(AmplitudeThreshold, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class ConvBlock(tf.keras.layers.Layer):
    """Block of convolutional layers. Implementing as a layer to make the summary nicer."""

    def __init__(self, a_min=0.0, a_max=1.0, **kwargs):
        raise NotImplementedError
        super(ConvBlock, self).__init__(**kwargs)
        if a_min < 0 and a_max <= a_min:
            raise ValueError("The supplied thresholds have to be 0 <= a_min < a_max.")
        self.a_min = a_min
        self.a_max = a_max

    def call(self, inputs):
        amps = tf.math.abs(inputs)
        amps_clipped = tf.clip_by_value(amps, self.a_min, self.a_max)
        outputs = inputs * tf.cast(amps_clipped / (amps + 1e-8), tf.complex64)
        return outputs

    def get_config(self):
        config = {"a_min": float(self.a_min), "a_max": float(self.a_max)}
        base_config = super(AmplitudeThreshold, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
