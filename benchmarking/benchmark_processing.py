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
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

from transformers import AutoImageProcessor

NUM_RUNS = 100
WARMUP_RUNS = 10
BENCHMARK_OUTPUT_FOLDER = "benchmark_outputs/json"

# fix random seed for reproducibility
np.random.seed(0)


def collate_fn(batch):
    data = {}
    # images = [Image.open(x["image_path"]).convert("RGB") for x in batch]
    images = [
        v2.functional.grayscale_to_rgb_image(read_image(x["image_path"])) for x in batch
    ]
    data["images"] = images
    annotations = []
    for x in batch:
        boxes = x["objects"]["bbox"]
        # convert to xyxy format
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
        labels = x["objects"]["category_id"]
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        annotations.append({"boxes": boxes, "labels": labels})
    data["original_size"] = [(x["height"], x["width"]) for x in batch]
    data["annotations"] = annotations
    return data


def collate_fn_PIL(batch):
    data = {}
    images = [Image.open(x["image_path"]).convert("RGB") for x in batch]
    # images = [
    #     v2.functional.grayscale_to_rgb_image(read_image(x["image_path"])) for x in batch
    # ]
    data["images"] = images
    annotations = []
    for x in batch:
        boxes = x["objects"]["bbox"]
        # convert to xyxy format
        boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
        labels = x["objects"]["category_id"]
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        annotations.append({"boxes": boxes, "labels": labels})
    data["original_size"] = [(x["height"], x["width"]) for x in batch]
    data["annotations"] = annotations
    return data


def get_random_image(size=(1920, 1080)):
    return np.random.randint(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)


def get_mean_med_max_diff(image1, image2):
    output_dict = {}
    if isinstance(image1, torch.Tensor):
        image1 = image1.cpu().numpy()
    output_dict["dtype"] = str(image1.dtype)
    mean = np.mean(np.abs(np.array(image1) - np.array(image2)))
    median = np.median(np.abs(np.array(image1) - np.array(image2)))
    max_diff = np.max(np.abs(np.array(image1) - np.array(image2)))
    output_dict["mean"] = mean.astype(float)
    output_dict["median"] = median.astype(float)
    output_dict["max"] = max_diff.astype(float)
    return output_dict


def benchmark_resize(image: np.ndarray, size=(640, 480)):
    # benchmark resize operation using bilinear resampling and assuming that the image is in a compatible format
    times = {}
    diffs = {}

    # resize using pillow (PIL)
    image_pil = PIL.Image.fromarray(image)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_pil_resized = image_pil.resize(size[::-1], resample=PIL.Image.BILINEAR)
    end = time.time()
    times["PIL"] = (end - start) / NUM_RUNS
    assert image_pil_resized.size == size[::-1]
    image_pil_resized = np.array(image_pil_resized).astype(np.int16)

    # resize using opencv
    start = time.time()
    for _ in range(NUM_RUNS):
        image_resized = cv2.resize(image, size[::-1], interpolation=cv2.INTER_LINEAR)
    end = time.time()
    times["OpenCV"] = (end - start) / NUM_RUNS
    assert image_resized.shape[:2] == size
    # convert to rgb to match PIL
    diffs["OpenCV"] = get_mean_med_max_diff(image_resized, image_pil_resized)

    image_pil_resized = np.array(image_pil_resized).transpose(2, 0, 1)
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
    diffs["Torchvision v1 cpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v2 transforms cpu
    transform_v2 = v2.Resize(size, interpolation=F.InterpolationMode.BILINEAR)
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 cpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v2 cpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v1 transforms gpu
    image_tensor.to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v1 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v2 transforms gpu
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v2 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )
    # resize using torchvision v1 on float32 tensor
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 float32 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v1 float32 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v2 on float32 tensor
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 float32 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v2 float32 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v1 on float16 tensor
    image_tensor = torch.tensor(image, dtype=torch.float16).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v1(image_tensor)
    end = time.time()
    times["Torchvision v1 float16 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v1 float16 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using torchvision v2 on float16 tensor
    image_tensor = torch.tensor(image, dtype=torch.float16).permute(2, 0, 1).to("cuda")
    start = time.time()
    for _ in range(NUM_RUNS):
        image_tensor_resized = transform_v2(image_tensor)
    end = time.time()
    times["Torchvision v2 float16 gpu"] = (end - start) / NUM_RUNS
    assert image_tensor_resized.cpu().numpy().shape[-2:] == size
    diffs["Torchvision v2 float16 gpu"] = get_mean_med_max_diff(
        image_tensor_resized, image_pil_resized
    )

    # resize using albumentations
    transform_albumentations = A.Resize(
        size[0], size[1], interpolation=cv2.INTER_LINEAR
    )
    start = time.time()
    for _ in range(NUM_RUNS):
        image_albumentations_resized = transform_albumentations(image=image)["image"]
    end = time.time()
    times["Albumentations"] = (end - start) / NUM_RUNS
    assert image_albumentations_resized.shape[:2][::-1] == size[::-1]
    image_pil_resized = np.array(image_pil_resized).transpose(1, 2, 0)
    diffs["Albumentations"] = get_mean_med_max_diff(
        image_albumentations_resized, image_pil_resized
    )
    return times, diffs


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
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Transformers"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    # Transformers image processor
    processor = AutoImageProcessor.from_pretrained(checkpoint, do_pad=False)
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = processor(image_tensor, return_tensors="pt").to(device)
        processing_time += time.time() - start_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Transformers (tensor inputs)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    # Transformers fast image processor
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, do_pad=False, use_fast=True
    )
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image = Image.open(image_path)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = processor(image, return_tensors="pt", device=device).to(
            device
        )
        processing_time += time.time() - start_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Transformers fast"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    processor = AutoImageProcessor.from_pretrained(
        checkpoint, do_pad=False, use_fast=True
    )
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = processor(
            image_tensor, return_tensors="pt", device=device
        ).to(device)
        processing_time += time.time() - start_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Transformers fast (tensor inputs)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    # processor = ViTImageProcessorFast.from_pretrained(checkpoint, do_pad=False)
    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image = Image.open(image_path)
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed = processor(image, return_tensors="pt").to(device)
    #     processing_time += time.time() - start_process
    # end = time.time()
    # times["Transformers Fast"] = {
    #     "total": (end - start) / NUM_RUNS,
    #     "loading": loading_time / NUM_RUNS,
    #     "processing": processing_time / NUM_RUNS,
    # }

    # processor = ViTImageProcessorFast.from_pretrained(checkpoint, do_pad=False)
    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed = processor(image_tensor, return_tensors="pt").to(device)
    #     processing_time += time.time() - start_process
    # end = time.time()
    # times["Transformers Fast (tensor inputs)"] = {
    #     "total": (end - start) / NUM_RUNS,
    #     "loading": loading_time / NUM_RUNS,
    #     "processing": processing_time / NUM_RUNS,
    # }

    optim_processor = BaseImageProcessorFast(**(processor.to_dict()))
    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed_optim = optim_processor(image_tensor)
        processing_time += time.time() - start_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Optim (uint8)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
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
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Optim (float32)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    start = time.time()
    loading_time = 0
    processing_time = 0
    for i in range(NUM_RUNS):
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
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
    end = time.time()
    times["Optim (float16)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    }

    return times


def benchmark_processor_batched(dataset, batch_size: int, checkpoint: str, device: str):
    times = {}
    # Transformers image processor
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_PIL)
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    start = time.time()
    total_loading_time = 0
    processing_time = 0
    total_runs = 0
    start_loading_time = time.time()
    for i, batch in enumerate(dataloader):
        total_loading_time += time.time() - start_loading_time
        start_process = time.time()
        images_processed = processor(batch["images"], return_tensors="pt").to(device)
        processing_time += time.time() - start_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            total_runs = 0
            total_loading_time = 0
            processing_time = 0
        start_loading_time = time.time()
    end = time.time()
    times["Transformers PIL"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": total_loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
    }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    processor = AutoImageProcessor.from_pretrained(checkpoint)
    start = time.time()
    total_loading_time = 0
    processing_time = 0
    total_runs = 0
    start_loading_time = time.time()
    for i, batch in enumerate(dataloader):
        total_loading_time += time.time() - start_loading_time
        start_process = time.time()
        images_processed = processor(batch["images"], return_tensors="pt").to(device)
        processing_time += time.time() - start_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            total_runs = 0
            total_loading_time = 0
            processing_time = 0
        start_loading_time = time.time()
    end = time.time()
    times["Transformers tensors"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": total_loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
    }

    # Transformers fast image processor
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_PIL)
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    start = time.time()
    total_loading_time = 0
    processing_time = 0
    total_runs = 0
    start_loading_time = time.time()
    for i, batch in enumerate(dataloader):
        total_loading_time += time.time() - start_loading_time
        start_process = time.time()
        images_processed = processor(
            batch["images"], return_tensors="pt", device=device
        ).to(device)
        processing_time += time.time() - start_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            total_runs = 0
            total_loading_time = 0
            processing_time = 0
        start_loading_time = time.time()

    end = time.time()
    times["Transformers fast PIL"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": total_loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
    }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    start = time.time()
    total_loading_time = 0
    processing_time = 0
    total_runs = 0
    start_loading_time = time.time()
    for i, batch in enumerate(dataloader):
        total_loading_time += time.time() - start_loading_time
        start_process = time.time()
        images_processed = processor(
            batch["images"], return_tensors="pt", device=device
        ).to(device)
        processing_time += time.time() - start_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            total_runs = 0
            total_loading_time = 0
            processing_time = 0
        start_loading_time = time.time()

    end = time.time()
    times["Transformers fast tensors"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": total_loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
    }

    return times


if __name__ == "__main__":
    # image = get_random_image(size=(480, 640))
    # size = (224, 224)
    # times_resize, diffs_resize = benchmark_resize(image, size)
    # mean = (0.485, 0.456, 0.406)
    # std = (0.229, 0.224, 0.225)
    # image_processing_ops_resize = {
    #     "small": {
    #         "times": times_resize,
    #         "size": size,
    #         "size_original": (480, 640),
    #         "diffs": diffs_resize,
    #     },
    # }
    # print(diffs_resize)
    # size = (800, 1333)
    # times_resize, diffs_resize = benchmark_resize(image, size)
    # image_processing_ops_resize["large"] = {
    #     "times": times_resize,
    #     "size": size,
    #     "size_original": (480, 640),
    #     "diffs": diffs_resize,
    # }

    # times_normalize = benchmark_normalize(image, mean, std)
    # image_processing_ops_normalize = {
    #     "times": times_normalize,
    #     "mean": mean,
    #     "std": std,
    # }

    # image_processing_ops = {
    #     "resize": image_processing_ops_resize,
    #     "normalize": image_processing_ops_normalize,
    # }

    # with open(f"{BENCHMARK_OUTPUT_FOLDER}/image_processing_ops_diffs.json", "w") as f:
    #     json.dump(image_processing_ops, f, indent=4)

    path = "/home/ubuntu/models_implem/000000039769.jpg"
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

    checkpoint = "facebook/detr-resnet-50"
    # checkpoint = "google/owlvit-base-patch32"

    device = "cuda"
    times_cuda = benchmark_processor(path, checkpoint, device)
    device = "cpu"
    times_cpu = benchmark_processor(path, checkpoint, device)
    processor_benchmark = {
        checkpoint: {
            "cuda": times_cuda,
            "cpu": times_cpu,
        }
    }

    # val_data = datasets.load_dataset(
    #     "yonigozlan/coco_detection_dataset_script",
    #     "2017",
    #     data_dir="/home/ubuntu/data",
    #     trust_remote_code=True,
    #     split="validation[:10%]",
    # )

    # device = "cuda"
    # times_cuda = benchmark_processor_batched(val_data, 8, checkpoint, device)
    # device = "cpu"
    # times_cpu = benchmark_processor_batched(val_data, 8, checkpoint, device)
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

    with open(
        f"{BENCHMARK_OUTPUT_FOLDER}/processor_detr_fast_benchmark_v2_test.json",
        "w",
    ) as f:
        json.dump(processor_benchmark, f, indent=4)
