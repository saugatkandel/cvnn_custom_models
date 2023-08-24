import dataclasses as dt

@dt.dataclass
class ConvParams:
    kernel_size: tuple = (3, 3)
    strides: tuple = (1, 1)
    padding: str = "same"
    data_format: str = "channels_last"


@dt.dataclass
class UpConvParams(ConvParams):
    strides: tuple = (2, 2)


@dt.dataclass
class DownConvParams(ConvParams):
    strides: tuple = (2, 2)
