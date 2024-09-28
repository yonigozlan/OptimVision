import json
import time

import albumentations as A
import cv2
import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F
from image_processing_fast import BaseImageProcessorFast
from PIL import Image
from torchvision.transforms import v2

from transformers import AutoImageProcessor

NUM_RUNS = 1000
BENCHMARK_OUTPUT_FOLDER = "benchmark_outputs/json"


def get_random_image(size=(1920, 1080)):
    return np.random.randint(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)


def benchmark_resize(image: np.ndarray, size=(640, 480)):
    # benchmark resize operation using bilinear resampling and assuming that the image is in a compatible format
    times = {}

    # resize using pillow (PIL)
    image_pil = PIL.Image.fromarray(image)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_pil_resized = image_pil.resize(size, resample=PIL.Image.BILINEAR)
    end = time.time()
    times["PIL"] = (end - start) / NUM_RUNS
    assert image_pil_resized.size == size

    # resize using opencv
    start = time.time()
    for _ in range(NUM_RUNS):
        image_resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    end = time.time()
    times["OpenCV"] = (end - start) / NUM_RUNS
    assert image_resized.shape[:2][::-1] == size

    # resize using torchvision v1 transforms cpu
    transform_v1 = torchvision.transforms.Resize(
        size, interpolation=F.InterpolationMode.BILINEAR
    )
    image_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1).to("cpu")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 cpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v2 transforms cpu
    transform_v2 = v2.Resize(size, interpolation=F.InterpolationMode.BILINEAR)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 cpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v1 transforms gpu
    image_tensor.to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v2 transforms gpu
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v1 on float32 tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 float32 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v2 on float32 tensor
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 float32 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using torchvision v1 on float16 tensor
    image_tensor = torch.tensor(image, dtype=torch.float16).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 float16 gpu"] = (end - start) / NUM_RUNS

    # resize using torchvision v2 on float16 tensor
    image_tensor = torch.tensor(image, dtype=torch.float16).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 float16 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size

    # resize using albumentations
    transform_albumentations = A.Resize(
        size[1], size[0], interpolation=cv2.INTER_LINEAR
    )
    start = time.time()
    for _ in range(NUM_RUNS):
        image_albumentations_resized = transform_albumentations(image=image)["image"]
    end = time.time()
    times["Albumentations"] = (end - start) / NUM_RUNS
    assert image_albumentations_resized.shape[:2][::-1] == size

    return times


def benchmark_normalize(image: np.ndarray, mean: tuple, std: tuple):
    times = {}

    # normalize using numpy
    mean_array = np.array(mean, dtype=np.float32)
    std_array = np.array(std, dtype=np.float32)
    image_numpy = image.astype(np.float32) / 255.0
    start = time.time()
    for _ in range(NUM_RUNS):
        image_numpy_normalized = (image_numpy - mean_array) / std_array
    end = time.time()
    times["Numpy"] = (end - start) / NUM_RUNS

    # normalize using albumentations
    transform_albumentations = A.Normalize(mean=mean, std=std)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_albumentations_normalized = transform_albumentations(image=image_numpy)[
            "image"
        ]
    end = time.time()
    times["Albumentations"] = (end - start) / NUM_RUNS

    # normalize using torchvision v1 transforms cpu
    transform_v1 = torchvision.transforms.Normalize(mean, std)
    image_tensor = (
        torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).to("cpu") / 255.0
    )
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_normalized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 cpu"] = (end - start) / NUM_RUNS

    # normalize using torchvision v2 transforms cpu
    transform_v2 = v2.Normalize(mean, std)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_normalized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 cpu"] = (end - start) / NUM_RUNS

    # normalize using torchvision v1 transforms gpu
    image_tensor = image_tensor.to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_normalized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 gpu"] = (end - start) / NUM_RUNS

    # normalize using torchvision v2 transforms gpu
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_normalized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 gpu"] = (end - start) / NUM_RUNS

    return times


def benchmark_load_from_path_to_tensor_gpu(path: str):
    times = {}

    # load image using PIL
    start = time.time()
    for _ in range(NUM_RUNS):
        image_pil = PIL.Image.open(path)
        image_tensor = v2.functional.pil_to_tensor(image_pil).to("cuda")
    end = time.time()
    times["PIL"] = (end - start) / NUM_RUNS

    # load image using opencv
    start = time.time()
    for _ in range(NUM_RUNS):
        image_opencv = cv2.imread(path)
        image_tensor = F.to_tensor(image_opencv).to("cuda")
    end = time.time()
    times["OpenCV"] = (end - start) / NUM_RUNS

    # load image using torchvision
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor = torchvision.io.read_image(path).to("cuda")
    end = time.time()
    times["torchvision.io.read_image"] = (end - start) / NUM_RUNS

    # # load image using albumentations
    # start = time.time()
    # for _ in range(NUM_RUNS):
    #     image_albumentations = A.load(path)
    # end = time.time()
    # times["Albumentations"] = (end - start) / NUM_RUNS

    return times


def benchmark_change_dtype(dtype: torch.dtype):
    times = {}
    path = "/home/ubuntu/models_implem/000000039769.jpg"
    # image = get_random_image(size=(1920, 1080))
    # image_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1).to("cuda")
    # change dtype using to()
    image_tensor = torchvision.io.read_image(path).unsqueeze(0)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_changed = image_tensor.to("cuda", dtype=dtype)
    end = time.time()
    print(image_tensor.dtype)
    times["tensor.to('cuda', dtype=dtype)"] = (end - start) / NUM_RUNS

    # change dtype using torchvision v2 transforms gpu
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_changed = image_tensor.to("cuda").to(dtype=dtype)
    end = time.time()
    print(image_tensor.dtype)
    times["tensor.to('cuda').to(dtype=dtype)"] = (end - start) / NUM_RUNS

    return times


def benchmark_processor(image_path: str, checkpoint: str, device: str):
    # Transformers image processor
    times = {}
    processor = AutoImageProcessor.from_pretrained(checkpoint, do_pad=False)
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image = Image.open(image_path)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = processor(image, return_tensors="pt").to(device)
        processing_time += time.time() - start_process
    end = time.time()
    times["Transformers"] = {
        "total": (end - start) / NUM_RUNS,
        "loading": loading_time / NUM_RUNS,
        "processing": processing_time / NUM_RUNS,
    }

    optim_processor = BaseImageProcessorFast(**(processor.to_dict()))
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(1000):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed_optim = optim_processor(image_tensor)
        processing_time += time.time() - start_process
    end = time.time()
    times["Optim (uint8)"] = {
        "total": (end - start) / NUM_RUNS,
        "loading": loading_time / NUM_RUNS,
        "processing": processing_time / NUM_RUNS,
    }

    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(1000):
        start_loadimage = time.time()
        image_tensor = (
            torchvision.io.read_image(image_path)
            .unsqueeze(0)
            .to(device)
            .to(torch.float32)
        )
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed_optim = optim_processor(image_tensor)
        processing_time += time.time() - start_process
    end = time.time()
    times["Optim (float32)"] = {
        "total": (end - start) / NUM_RUNS,
        "loading": loading_time / NUM_RUNS,
        "processing": processing_time / NUM_RUNS,
    }

    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(1000):
        start_loadimage = time.time()
        image_tensor = (
            torchvision.io.read_image(image_path)
            .unsqueeze(0)
            .to(device)
            .to(torch.float16)
        )
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed_optim = optim_processor(image_tensor)
        processing_time += time.time() - start_process
    end = time.time()
    times["Optim (float16)"] = {
        "total": (end - start) / NUM_RUNS,
        "loading": loading_time / NUM_RUNS,
        "processing": processing_time / NUM_RUNS,
    }

    return times


if __name__ == "__main__":
    image = get_random_image(size=(480, 640))
    size = (224, 224)
    times_resize = benchmark_resize(image, size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    image_processing_ops_resize = {
        "small": {
            "times": times_resize,
            "size": size,
            "size_original": (480, 640),
        },
    }
    size = (800, 1333)
    times_resize = benchmark_resize(image, size)
    image_processing_ops_resize["large"] = {
        "times": times_resize,
        "size": size,
        "size_original": (480, 640),
    }

    times_normalize = benchmark_normalize(image, mean, std)
    image_processing_ops_normalize = {
        "times": times_normalize,
        "mean": mean,
        "std": std,
    }

    image_processing_ops = {
        "resize": image_processing_ops_resize,
        "normalize": image_processing_ops_normalize,
    }

    with open(f"{BENCHMARK_OUTPUT_FOLDER}/image_processing_ops.json", "w") as f:
        json.dump(image_processing_ops, f, indent=4)

    # path = "/home/ubuntu/models_implem/000000039769.jpg"
    # times = benchmark_load_from_path_to_tensor_gpu(path)
    # load_from_path_to_tensor_gpu = {
    #     "load_from_path_to_tensor_gpu": times,
    # }
    # with open("load_from_path_to_tensor_gpu.json", "w") as f:
    #     json.dump(load_from_path_to_tensor_gpu, f, indent=4)

    # times_float32 = benchmark_change_dtype(dtype=torch.float32)
    # times_float16 = benchmark_change_dtype(dtype=torch.float16)
    # change_dtype = {
    #     "float32": times_float32,
    #     "float16": times_float16,
    # }
    # with open(f"{BENCHMARK_OUTPUT_FOLDER}/change_dtype_from_uint8_2.json", "w") as f:
    #     json.dump(change_dtype, f, indent=4)

    # checkpoint = "facebook/detr-resnet-50"
    # device = "cuda"
    # times_cuda = benchmark_processor(path, checkpoint, device)
    # device = "cpu"
    # times_cpu = benchmark_processor(path, checkpoint, device)
    # processor_benchmark = {
    #     checkpoint: {
    #         "cuda": times_cuda,
    #         "cpu": times_cpu,
    #     }
    # }

    # checkpoint = "PekingU/rtdetr_r101vd"
    # device = "cuda"
    # times_cuda = benchmark_processor(path, checkpoint, device)
    # device = "cpu"
    # times_cpu = benchmark_processor(path, checkpoint, device)
    # processor_benchmark[checkpoint] = {
    #     "cuda": times_cuda,
    #     "cpu": times_cpu,
    # }

    # with open(f"{BENCHMARK_OUTPUT_FOLDER}/processor_benchmark.json", "w") as f:
    #     json.dump(processor_benchmark, f, indent=4)
