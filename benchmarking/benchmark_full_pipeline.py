import json
import time

import torch
import torchvision
from image_processing_fast import BaseImageProcessorFast
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForObjectDetection

NUM_RUNS = 100
BENCHMARK_OUTPUT_FOLDER = "benchmark_outputs/json"
WARMUP_RUNS = 10


def benchmark_processor(
    image_path: str,
    checkpoint: str,
    device: str,
    compiled: bool = False,
    dtype=torch.float32,
):
    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).to(dtype)
    if compiled:
        model = torch.compile(model, mode="reduce-overhead")

    # Transformers image processor
    times = {}
    processor = AutoImageProcessor.from_pretrained(checkpoint, do_pad=False)
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image = Image.open(image_path)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = processor(image, return_tensors="pt").to(device).to(dtype)
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0

    end = time.time()
    times["Transformers"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    optim_processor = BaseImageProcessorFast(**(processor.to_dict()))
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed_optim = optim_processor(image_tensor).to(dtype)
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed_optim)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
    end = time.time()
    times["Optim (uint8)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
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
        images_processed_optim = optim_processor(image_tensor, dtype=torch.float32).to(
            dtype
        )
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed_optim)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
    end = time.time()
    times["Optim (float32)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    model = (
        AutoModelForObjectDetection.from_pretrained(checkpoint)
        .to(device)
        .to(dtype=torch.float16)
    )
    if compiled:
        model = torch.compile(model, mode="reduce-overhead")

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
        images_processed_optim = optim_processor(image_tensor, dtype=torch.float16).to(
            dtype
        )
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed_optim)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
    end = time.time()
    times["Optim (float16)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    return times


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "/home/ubuntu/models_implem/000000039769.jpg"
    checkpoint = "facebook/detr-resnet-50"

    times = benchmark_processor(
        path, checkpoint, device, compiled=False, dtype=torch.float16
    )
    full_pipeline = {
        "eager": times,
    }
    times = benchmark_processor(
        path, checkpoint, device, compiled=True, dtype=torch.float16
    )
    full_pipeline["compiled"] = times

    with open(
        f"{BENCHMARK_OUTPUT_FOLDER}/benchmark_results_full_pipeline_detr.json",
        "w",
    ) as file:
        json.dump(full_pipeline, file, indent=4)
