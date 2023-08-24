import tensorflow as tf

from keras.engine import data_adapter
from .complex_models import conv_fn, up_conv_fn, block_fn
from .params import ConvParams, UpConvParams, DownConvParams

DEFAULT_ACTIVATION = "cleakyrelu"


def create_model_64_to_128(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomy for some data/batch sizes sometimes"""

    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")
    down_stack = [input_img]
    for ix, nf in enumerate([32, 64, 128]):
        down_layer = None if ix == 0 else 0
        down_stack += [
            *block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_down_stride_in_layer=down_layer,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    up_stack = block_fn(
        128,
        down_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        activation=activation,
        name="btn",
    )

    for ix, nf in enumerate([64, 32, 16]):
        up_stack += [
            *block_fn(
                nf,
                up_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=activation,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]

    out_layer = conv_fn(1, conv_params=conv_params, name=f"l_{ix}", activation="linear")(up_stack[-1])
    # model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model

def create_model_128_to_128(
    img_h: int = 128,
    img_w: int = 128,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomy for some data/batch sizes sometimes"""

    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")
    down_stack = [input_img]
    for ix, nf in enumerate([16, 32, 64, 128]):
        down_layer = None if ix == 0 else 0
        down_stack += [
            *block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_down_stride_in_layer=down_layer,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    up_stack = block_fn(
        128,
        down_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        activation=activation,
        name="btn",
    )

    for ix, nf in enumerate([64, 32, 16]):
        up_stack += [
            *block_fn(
                nf,
                up_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=activation,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]

    out_layer = conv_fn(1, conv_params=conv_params, name=f"l_{ix}", activation="linear")(up_stack[-1])
    # model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model


def create_model_256_to_256(
    img_h: int = 256,
    img_w: int = 256,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomy for some data/batch sizes sometimes"""

    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")
    down_stack = [input_img]
    for ix, nf in enumerate([16, 32, 64, 128, 256]):
        down_layer = None if ix == 0 else 0
        down_stack += [
            *block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_down_stride_in_layer=down_layer,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    up_stack = block_fn(
        256,
        down_stack[-1],
        use_down_stride_in_layer=0,
        use_up_stride_in_layer=1,
        conv_params=conv_params,
        down_conv_params=down_conv_params,
        activation=activation,
        name="btn",
    )

    for ix, nf in enumerate([128, 64, 32, 16]):
        up_stack += [
            *block_fn(
                nf,
                up_stack[-1],
                use_up_stride_in_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=activation,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]

    out_layer = conv_fn(1, conv_params=conv_params, name=f"l_{ix}", activation="linear")(up_stack[-1])
    # model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model


def create_model_64_to_128_pool_upsamp(
    img_h: int = 64,
    img_w: int = 64,
    conv_params: ConvParams = None,
    up_conv_params: UpConvParams = None,
    down_conv_params: DownConvParams = None,
    activation: str = DEFAULT_ACTIVATION,
    apply_batchnorm: bool = False,
) -> tf.keras.Model:
    """Using up_stride True by default because argmax is problematic and crashes randomy for some data/batch sizes sometimes"""

    input_img = tf.keras.Input(shape=(img_h, img_w, 1), dtype="complex64")
    down_stack = [input_img]
    for ix, nf in enumerate([32, 64, 128]):
        down_layer = 1
        down_stack += [
            *block_fn(
                n_filters=nf,
                l0=down_stack[-1],
                use_pool_after_layer=down_layer,
                conv_params=conv_params,
                down_conv_params=down_conv_params,
                name=f"dl_{ix}",
                activation=activation,
                apply_batchnorm=apply_batchnorm,
            )
        ]

    up_stack = block_fn(
        128,
        down_stack[-1],
        use_upsamp_after_layer=1,
        conv_params=conv_params,
        up_conv_params=up_conv_params,
        activation=activation,
        name="btn",
    )

    for ix, nf in enumerate([64, 32, 16]):
        up_stack += [
            *block_fn(
                nf,
                up_stack[-1],
                use_upsamp_after_layer=1,
                conv_params=conv_params,
                up_conv_params=up_conv_params,
                activation=activation,
                name=f"ul_{ix}",
                n_layers=2,
            )
        ]

    out_layer = conv_fn(1, conv_params=conv_params, name=f"ul_{ix}", activation="linear")(up_stack[-1])
    model = tf.keras.Model(inputs=[input_img], outputs=[out_layer])
    return model
