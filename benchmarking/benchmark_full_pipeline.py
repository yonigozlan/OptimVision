import json
import time

import datasets
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2

from transformers import AutoImageProcessor, AutoModelForObjectDetection

NUM_RUNS = 100
BENCHMARK_OUTPUT_FOLDER = "benchmark_outputs/json"
WARMUP_RUNS = 50


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
    times = {}

    # Transformers image processor
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
    times["(Current Processor)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    processor = AutoImageProcessor.from_pretrained(
        checkpoint, do_pad=False, use_fast=True
    )
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    for i in range(NUM_RUNS):
        start_loadimage = time.time()
        image_tensor = torchvision.io.read_image(image_path).unsqueeze(0)
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = (
            processor(image_tensor, return_tensors="pt", device="cpu")
            .to(device)
            .to(dtype)
        )
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
    times["Fast (CPU Processing)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    processor = AutoImageProcessor.from_pretrained(
        checkpoint, do_pad=False, use_fast=True
    )
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
        images_processed = processor(
            image_tensor, return_tensors="pt", device=device
        ).to(dtype)
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
    times["Fast (CUDA Processing)"] = {
        "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
        "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
        "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
        "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
        "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    }

    # optim_processor = BaseImageProcessorFast(**(processor.to_dict()))
    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # inference_time = 0
    # post_process_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image_tensor = torchvision.io.read_image(image_path).unsqueeze(0).to(device)
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed_optim = optim_processor(image_tensor)
    #     if images_processed_optim.pixel_values.dtype != dtype:
    #         images_processed_optim = images_processed_optim.to(dtype)
    #     processing_time += time.time() - start_process
    #     start_inference = time.time()
    #     with torch.no_grad():
    #         outputs = model(**images_processed_optim)
    #     _ = outputs[0].cpu()
    #     end_inference = time.time()
    #     inference_time += end_inference - start_inference
    #     start_post_process = time.time()
    #     processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
    #     end_post_process = time.time()
    #     post_process_time += end_post_process - start_post_process
    #     if i == WARMUP_RUNS:
    #         start = time.time()
    #         loading_time = 0
    #         processing_time = 0
    #         inference_time = 0
    #         post_process_time = 0
    # end = time.time()
    # times["Optim (uint8)"] = {
    #     "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
    #     "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
    #     "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    #     "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
    #     "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    # }

    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image_tensor = (
    #         torchvision.io.read_image(image_path)
    #         .unsqueeze(0)
    #         .to(device)
    #         .to(torch.float32)
    #     )
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed_optim = optim_processor(image_tensor, dtype=torch.float32)
    #     if images_processed_optim.pixel_values.dtype != dtype:
    #         images_processed_optim = images_processed_optim.to(dtype)
    #     processing_time += time.time() - start_process
    #     start_inference = time.time()
    #     with torch.no_grad():
    #         outputs = model(**images_processed_optim)
    #     _ = outputs[0].cpu()
    #     end_inference = time.time()
    #     inference_time += end_inference - start_inference
    #     start_post_process = time.time()
    #     processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
    #     end_post_process = time.time()
    #     post_process_time += end_post_process - start_post_process
    #     if i == WARMUP_RUNS:
    #         start = time.time()
    #         loading_time = 0
    #         processing_time = 0
    #         inference_time = 0
    #         post_process_time = 0
    # end = time.time()
    # times["Optim (float32)"] = {
    #     "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
    #     "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
    #     "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    #     "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
    #     "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    # }

    # model = (
    #     AutoModelForObjectDetection.from_pretrained(checkpoint)
    #     .to(device)
    #     .to(dtype=torch.float16)
    # )
    # if compiled:
    #     model = torch.compile(model, mode="reduce-overhead")

    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image_tensor = (
    #         torchvision.io.read_image(image_path)
    #         .unsqueeze(0)
    #         .to(device)
    #         .to(torch.float16)
    #     )
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed_optim = optim_processor(
    #         image_tensor, dtype=torch.float16, processing_dtype=torch.uint8
    #     )
    #     if images_processed_optim.pixel_values.dtype != dtype:
    #         images_processed_optim = images_processed_optim.to(dtype)
    #     processing_time += time.time() - start_process
    #     start_inference = time.time()
    #     with torch.no_grad():
    #         outputs = model(**images_processed_optim)
    #     _ = outputs[0].cpu()
    #     end_inference = time.time()
    #     inference_time += end_inference - start_inference
    #     start_post_process = time.time()
    #     processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
    #     end_post_process = time.time()
    #     post_process_time += end_post_process - start_post_process
    #     if i == WARMUP_RUNS:
    #         start = time.time()
    #         loading_time = 0
    #         processing_time = 0
    #         inference_time = 0
    #         post_process_time = 0
    # end = time.time()
    # times["Optim (float16)"] = {
    #     "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
    #     "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
    #     "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    #     "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
    #     "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    # }

    # start = time.time()
    # loading_time = 0
    # processing_time = 0
    # for i in range(NUM_RUNS):
    #     start_loadimage = time.time()
    #     image_tensor = (
    #         torchvision.io.read_image(image_path)
    #         .unsqueeze(0)
    #         .to(device)
    #         .to(torch.float16)
    #     )
    #     loading_time += time.time() - start_loadimage
    #     start_process = time.time()
    #     images_processed_optim = optim_processor(image_tensor, dtype=torch.float16)
    #     if images_processed_optim.pixel_values.dtype != dtype:
    #         images_processed_optim = images_processed_optim.to(dtype)
    #     processing_time += time.time() - start_process
    #     start_inference = time.time()
    #     with torch.no_grad():
    #         outputs = model(**images_processed_optim)
    #     _ = outputs[0].cpu()
    #     end_inference = time.time()
    #     inference_time += end_inference - start_inference
    #     start_post_process = time.time()
    #     processor.post_process_object_detection(outputs, target_sizes=[(480, 640)])
    #     end_post_process = time.time()
    #     post_process_time += end_post_process - start_post_process
    #     if i == WARMUP_RUNS:
    #         start = time.time()
    #         loading_time = 0
    #         processing_time = 0
    #         inference_time = 0
    #         post_process_time = 0
    # end = time.time()
    # times["Optim (float16, float32 processing)"] = {
    #     "total": (end - start) / (NUM_RUNS - WARMUP_RUNS),
    #     "loading": loading_time / (NUM_RUNS - WARMUP_RUNS),
    #     "processing": processing_time / (NUM_RUNS - WARMUP_RUNS),
    #     "inference": inference_time / (NUM_RUNS - WARMUP_RUNS),
    #     "post_process": post_process_time / (NUM_RUNS - WARMUP_RUNS),
    # }

    return times


def benchmark_processor_batched(
    dataset,
    batch_size: int,
    checkpoint: str,
    device: str,
    compiled: bool = False,
    dtype=torch.float32,
    pad_size=None,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = AutoModelForObjectDetection.from_pretrained(checkpoint).to(device).to(dtype)
    if compiled:
        model = torch.compile(model, mode="reduce-overhead")
    times = {}

    # Transformers image processor
    processor = AutoImageProcessor.from_pretrained(checkpoint, pad_size=pad_size)
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    total_runs = 0
    start_loadimage = time.time()
    for i, batch in enumerate(dataloader):
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = (
            processor(batch["images"], return_tensors="pt").to(device).to(dtype)
        )
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(
            outputs,
            target_sizes=[
                batch["images"][i].shape[-2:] for i in range(len(batch["images"]))
            ],
        )
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            total_runs = 0
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
        start_loadimage = time.time()

    end = time.time()
    times["(Current Processor)"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
        "inference": inference_time / (total_runs - WARMUP_RUNS),
        "post_process": post_process_time / (total_runs - WARMUP_RUNS),
    }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, use_fast=True, pad_size=pad_size
    )
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    start_loadimage = time.time()
    total_runs = 0
    for i, batch in enumerate(dataloader):
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = (
            processor(batch["images"], return_tensors="pt", device="cpu")
            .to(device)
            .to(dtype)
        )

        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(
            outputs,
            target_sizes=[
                batch["images"][i].shape[-2:] for i in range(len(batch["images"]))
            ],
        )
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            total_runs = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
        start_loadimage = time.time()

    end = time.time()
    times["Fast (CPU Processing)"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
        "inference": inference_time / (total_runs - WARMUP_RUNS),
        "post_process": post_process_time / (total_runs - WARMUP_RUNS),
    }

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    processor = AutoImageProcessor.from_pretrained(
        checkpoint, use_fast=True, pad_size=pad_size
    )
    start = time.time()
    loading_time = 0
    processing_time = 0
    inference_time = 0
    post_process_time = 0
    start_loadimage = time.time()
    total_runs = 0
    for i, batch in enumerate(dataloader):
        loading_time += time.time() - start_loadimage
        start_process = time.time()
        images_processed = (
            processor(batch["images"], return_tensors="pt", device=device)
            .to(device)
            .to(dtype)
        )
        processing_time += time.time() - start_process
        start_inference = time.time()
        with torch.no_grad():
            outputs = model(**images_processed)
        _ = outputs[0].cpu()
        end_inference = time.time()
        inference_time += end_inference - start_inference
        start_post_process = time.time()
        processor.post_process_object_detection(
            outputs,
            target_sizes=[
                batch["images"][i].shape[-2:] for i in range(len(batch["images"]))
            ],
        )
        end_post_process = time.time()
        post_process_time += end_post_process - start_post_process
        total_runs += 1
        if i == WARMUP_RUNS:
            start = time.time()
            loading_time = 0
            processing_time = 0
            inference_time = 0
            post_process_time = 0
            total_runs = 0
        start_loadimage = time.time()

    end = time.time()
    times["Fast (CUDA Processing)"] = {
        "total": (end - start) / (total_runs - WARMUP_RUNS),
        "loading": loading_time / (total_runs - WARMUP_RUNS),
        "processing": processing_time / (total_runs - WARMUP_RUNS),
        "inference": inference_time / (total_runs - WARMUP_RUNS),
        "post_process": post_process_time / (total_runs - WARMUP_RUNS),
    }

    return times


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # path = "/home/ubuntu/models_implem/000000039769.jpg"
    # checkpoint = "PekingU/rtdetr_r50vd"
    # checkpoint = "facebook/detr-resnet-50"
    checkpoint = "SenseTime/deformable-detr"
    # times = benchmark_processor(
    #     path, checkpoint, device, compiled=False, dtype=torch.float32
    # )
    # full_pipeline = {
    #     "Eager": times,
    # }
    # times = benchmark_processor(
    #     path, checkpoint, device, compiled=True, dtype=torch.float32
    # )
    # full_pipeline["Compiled"] = times

    # with open(
    #     f"{BENCHMARK_OUTPUT_FOLDER}/benchmark_results_full_pipeline_deformable_detr_fast_single_image.json",
    #     "w",
    # ) as file:
    #     json.dump(full_pipeline, file, indent=4)

    val_data = datasets.load_dataset(
        "yonigozlan/coco_detection_dataset_script",
        "2017",
        data_dir="/home/ubuntu/data",
        trust_remote_code=True,
        split="validation[:10%]",
    )

    times = benchmark_processor_batched(
        val_data,
        8,
        checkpoint,
        device,
        compiled=False,
        dtype=torch.float16,
        pad_size={"width": 1333, "height": 1333},
    )
    full_pipeline = {
        "Eager": times,
    }
    times = benchmark_processor_batched(
        val_data,
        8,
        checkpoint,
        device,
        compiled=True,
        dtype=torch.float16,
        pad_size={"width": 1333, "height": 1333},
    )
    full_pipeline["Compiled"] = times

    with open(
        f"{BENCHMARK_OUTPUT_FOLDER}/benchmark_results_full_pipeline_deformable_detr_fast_batched_padded.json",
        "w",
    ) as file:
        json.dump(full_pipeline, file, indent=4)
