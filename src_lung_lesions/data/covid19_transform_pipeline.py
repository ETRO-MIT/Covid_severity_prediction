# Copyright 2021 Lucas Fidon
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    CropForegroundd,
    NormalizeIntensityd,
    CastToTyped,
    ToTensord,
)


def covid19_inference_transform(config, image_keys):
    inference_transform = Compose([
        LoadImaged(keys=image_keys),
        AddChanneld(keys=image_keys),
        CropForegroundd(keys=image_keys, source_key=image_keys[0]),
        NormalizeIntensityd(keys=image_keys, nonzero=False, channel_wise=True),
        CastToTyped(keys=image_keys, dtype=(np.float32,)),
        ToTensord(keys=image_keys),
    ])
    return inference_transform
