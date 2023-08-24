import tensorflow as tf
import dataclasses as dt
from .params import ConvParams, UpConvParams, DownConvParams
from typing import Tuple
from tensorflow.python.keras import activations

DEFAULT_REAL_ACTIVATION = "leaky_relu"


def conv_fn(n_filters, conv_params=None, activation=DEFAULT_REAL_ACTIVATION, name=None):
    if conv_params is None:
        conv_params = ConvParams()

    return tf.keras.layers.Conv2D(
        n_filters, activation=activation, **dt.asdict(conv_params), name=name
    )


def down_conv_fn(
    n_filters, down_conv_params=None, activation=DEFAULT_REAL_ACTIVATION, name=None
):
    if down_conv_params is None:
        down_conv_params = DownConvParams()

    return tf.keras.layers.Conv2D(
        n_filters, activation=activation, **dt.asdict(down_conv_params), name=name
    )


def up_conv_fn(
    n_filters, up_conv_params=None, activation=DEFAULT_REAL_ACTIVATION, name=None
):
    if up_conv_params is None:
        up_conv_params = UpConvParams()

    return tf.keras.layers.Conv2DTranspose(
        n_filters, activation=activation, **dt.asdict(up_conv_params), name=name
    )


def pool_fn(
    pool_size: Tuple = (2, 2),
    padding: str = "same",
    data_format: str = "channels_last",
    **kwargs,
):
    return tf.keras.layers.MaxPool2D(
        pool_size=pool_size, padding=padding, data_format=data_format, **kwargs
    )


def upsample_fn(
    size: Tuple = (2, 2),
    interpolation: str = "bilinear",
    data_format: str = "channels_last",
    **kwargs,
):
    return tf.keras.layers.UpSampling2D(
        size=size, interpolation=interpolation, data_format=data_format, **kwargs
    )


def block_fn(
    n_filters: int,
    l0: tf.keras.layers.Layer,
    use_down_stride_in_layer: int = None,
    use_up_stride_in_layer: int = None,
    use_pool_after_layer: int = None,
    use_upsamp_after_layer: int = None,
    apply_batchnorm: bool = False,
    n_layers: int = 2,
    activation: str = DEFAULT_REAL_ACTIVATION,
    conv_params: ConvParams = None,
    up_conv_params: ConvParams = None,
    down_conv_params: ConvParams = None,
    name=None,
):

    activation = activations.get(activation)

    if (use_down_stride_in_layer is not None) and (use_up_stride_in_layer is not None):
        if use_down_stride_in_layer == use_up_stride_in_layer:
            raise ValueError

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
            block += [
                tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(
                    block[-1]
                )
            ]

        block += [activation(block[-1])]

        l0 = block[-1]

        if use_pool_after_layer == ix:
            block += [pool_fn(name=f"{name}_pool")(l0)]
            l0 = block[-1]
        if use_upsamp_after_layer == ix:
            block += [upsample_fn(name=f"{name}_upsamp")(l0)]
            l0 = block[-1]

    # if apply_batchnorm:
    #    block += [tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(l0)]

    return block
