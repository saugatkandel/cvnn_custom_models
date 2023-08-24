import tensorflow as tf
from . import real_models as rmods
from . import complex_models as cmods
from .params import ConvParams, UpConvParams, DownConvParams


DEFAULT_REAL_ACTIVATION = "leaky_relu"
DEFAULT_COMPLEX_ACTIVATION = "cleakyrelu"


def create_model_64_to_128(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    real_activation: str = DEFAULT_REAL_ACTIVATION,
    complex_activation: str = DEFAULT_COMPLEX_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomly
    for some data/batch sizes sometimes"""

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

    up_stack = [tf.cast(down_stack[-1], tf.complex64)]
    up_stack += cmods.block_fn(
        128,
        up_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=complex_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn",
    )
    for ix, nf in enumerate([64, 32, 16]):
        up_stack += [
            *cmods.block_fn(
                nf,
                up_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=complex_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]
    out_layer = cmods.conv_fn(1, conv_params=None, name=f"ul_{ix}", activation="linear")(up_stack[-1])
    # model = cmods.CustomModelForJitCorrection(inputs=[input_img], outputs=[out_layer])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model


def create_model_64_to_128_reduced(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    real_activation: str = DEFAULT_REAL_ACTIVATION,
    complex_activation: str = DEFAULT_COMPLEX_ACTIVATION,
    apply_batchnorm=False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomly
    for some data/batch sizes sometimes"""

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
    up_stack = [tf.cast(down_stack[-1], tf.complex64)]
    up_stack += cmods.block_fn(
        96,
        up_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        up_conv_params=up_conv_params,
        activation=complex_activation,
        apply_batchnorm=apply_batchnorm,
        name="btn",
    )

    for ix, nf in enumerate([48, 24, 12]):
        up_stack += [
            *cmods.block_fn(
                nf,
                up_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=complex_activation,
                apply_batchnorm=apply_batchnorm,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]
    out_layer = cmods.conv_fn(1, conv_params=None, name=f"ul_{ix}", activation="linear")(up_stack[-1])
    # model = cmods.CustomModelForJitCorrection(inputs=[input_img], outputs=[out_layer])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model
