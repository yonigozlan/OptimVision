from typing import List, Tuple, Union

import numpy as np
import PIL
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from transformers.image_processing_base import BatchFeature

ImageInput = Union[
    "PIL.Image.Image",
    np.ndarray,
    "torch.Tensor",
    List["PIL.Image.Image"],
    List[np.ndarray],
    List["torch.Tensor"],
]  # noqa

InterpolationModeConverter = {
    0: InterpolationMode.NEAREST,
    1: InterpolationMode.LANCZOS,
    2: InterpolationMode.BILINEAR,
    3: InterpolationMode.BICUBIC,
    4: InterpolationMode.BOX,
    5: InterpolationMode.HAMMING,
}


def get_size_with_aspect_ratio(image_size, size, max_size=None) -> Tuple[int, int]:
    """
    Computes the output image size given the input image size and the desired output size.

    Args:
        image_size (`Tuple[int, int]`):
            The input image size.
        size (`int`):
            The desired output size.
        max_size (`int`, *optional*):
            The maximum allowed output size.
    """
    height, width = image_size
    raw_size = None
    if max_size is not None:
        min_original_size = float(min((height, width)))
        max_original_size = float(max((height, width)))
        if max_original_size / min_original_size * size > max_size:
            raw_size = max_size * min_original_size / max_original_size
            size = int(round(raw_size))

    if (height <= width and height == size) or (width <= height and width == size):
        oh, ow = height, width
    elif width < height:
        ow = size
        if max_size is not None and raw_size is not None:
            oh = int(raw_size * height / width)
        else:
            oh = int(size * height / width)
    else:
        oh = size
        if max_size is not None and raw_size is not None:
            ow = int(raw_size * width / height)
        else:
            ow = int(size * width / height)

    return (oh, ow)


class BaseImageProcessorFast:
    def __init__(self, **kwargs):
        self.kwargs = kwargs if kwargs else {}

    def __call__(self, images, **kwargs) -> BatchFeature:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    def preprocess(self, images: ImageInput, **kwargs) -> BatchFeature:
        self.kwargs.update(kwargs)

        do_rescale = self.kwargs.get("do_rescale", True)
        do_normalize = self.kwargs.get("do_normalize", False)
        do_resize = self.kwargs.get("do_resize", False)
        output_dtype = self.kwargs.get("output_dtype", torch.float32)
        processing_dtype = self.kwargs.get("processing_dtype", torch.float32)
        target_size = self.kwargs.get("size", (224, 224))
        resample = self.kwargs.get("resample", InterpolationMode.BILINEAR)
        images_list = []
        for image in images:
            if isinstance(resample, int):
                resample = InterpolationModeConverter[resample]
            if isinstance(target_size, dict):
                if "height" in target_size:
                    size = (target_size["height"], target_size["width"])
                elif "shortest_edge" in target_size:
                    # Resize the image so that the shortest edge or the longest edge is of the given size
                    # while maintaining the aspect ratio of the original image.
                    size = get_size_with_aspect_ratio(
                        image.size()[-2:],
                        target_size["shortest_edge"],
                        target_size["longest_edge"],
                    )
                else:
                    raise ValueError(
                        "size should be either (height, width) or (shortest_edge, longest_edge)"
                    )
            if processing_dtype != image.dtype:
                image = image.to(processing_dtype)
            orig_dtype = image.dtype
            image_mean = self.kwargs.get("image_mean", (0.485, 0.456, 0.406))
            image_std = self.kwargs.get("image_std", (0.229, 0.224, 0.225))

            pixel_values = image
            if do_resize:
                pixel_values = v2.functional.resize(image, size, interpolation=resample)
            pixel_values = v2.functional.to_dtype(
                pixel_values, output_dtype, scale=True
            )
            if do_rescale:
                if orig_dtype != torch.uint8:
                    pixel_values = pixel_values / 255.0
            if do_normalize:
                pixel_values = v2.functional.normalize(
                    pixel_values, mean=image_mean, std=image_std
                )

            images_list.append(pixel_values)

        return BatchFeature(data={"pixel_values": torch.stack(images_list, dim=0)})
