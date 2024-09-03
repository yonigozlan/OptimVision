import json
import time

import requests
import torch
from optim_deformable_detr import OptimDeformableDetrForObjectDetection
from optim_rt_detr import OptimRTDetrForObjectDetection
from PIL import Image

from transformers import (
    AutoProcessor,
    DeformableDetrForObjectDetection,
    DetrForObjectDetection,
    RTDetrForObjectDetection,
)

IMAGE_SIZE = (640, 640)
MANUAL_INFERENCE_STEPS = 100

MODEL_NAMES_MODEL_CORRENSPONDENCE = {
    "rt_detr": RTDetrForObjectDetection,
    "optim_rt_detr": OptimRTDetrForObjectDetection,
    "deformable_detr": DeformableDetrForObjectDetection,
    "optim_deformable_detr": OptimDeformableDetrForObjectDetection,
    "detr": DetrForObjectDetection,
}


MODEL_NAMES_WEIGHTS_CORRESPONDENCE = {
    "rt_detr": "PekingU/rtdetr_r50vd_coco_o365",
    "optim_rt_detr": "PekingU/rtdetr_r50vd_coco_o365",
    "deformable_detr": "SenseTime/deformable-detr",
    "optim_deformable_detr": "SenseTime/deformable-detr",
    "detr": "facebook/detr-resnet-50",
}


def get_sample_input_image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


def benchmark_model(model, processor, model_name, experiment):
    image = get_sample_input_image()
    inputs = processor(images=image, return_tensors="pt", size=IMAGE_SIZE).to("cuda")

    outputs = None
    results = None
    # Sequential pytorch/tensorboard profiling
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./benchmark/{model_name}",
            worker_name=f"{experiment}_sequential",
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.no_grad():
            for _ in range(10):
                profiler.step()
                outputs = model(**inputs)
                results = processor.post_process_object_detection(
                    outputs, target_sizes=[image.size[::-1]]
                )

    del outputs, results
    outputs = None
    results = None

    # Sequential manual profiling
    start_time = time.time()
    with torch.no_grad():
        for _ in range(MANUAL_INFERENCE_STEPS):
            outputs = model(**inputs)
            results = processor.post_process_object_detection(
                outputs, target_sizes=[image.size[::-1]]
            )
    end_time = time.time()
    average_inference_time_sequential = (end_time - start_time) / MANUAL_INFERENCE_STEPS
    average_fps_sequential = MANUAL_INFERENCE_STEPS / (end_time - start_time + 1e-6)

    del outputs, results
    outputs = None
    results = None

    # Only model inferences then all post-processing pytorch/tensorboard profiling
    outputs_list = []
    results_list = []
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./benchmark/{model_name}",
            worker_name=f"{experiment}_split",
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.no_grad():
            for _ in range(MANUAL_INFERENCE_STEPS):
                profiler.step()
                outputs = model(**inputs)
                outputs_list.append(outputs)
        for output in outputs_list:
            results = processor.post_process_object_detection(
                output, target_sizes=[image.size[::-1]]
            )
            results_list.append(results)

    del outputs, results
    outputs = None
    results_list = None

    # All model inferences then all post-processing manual profiling
    outputs_list = []
    results_list = []
    start_time = time.time()
    with torch.no_grad():
        for _ in range(MANUAL_INFERENCE_STEPS):
            outputs = model(**inputs)
            outputs_list.append(outputs)
    for output in outputs_list:
        results = processor.post_process_object_detection(
            output, target_sizes=[image.size[::-1]]
        )
        results_list.append(results)
    end_time = time.time()

    average_inference_time_split = (end_time - start_time) / MANUAL_INFERENCE_STEPS
    average_fps__split = MANUAL_INFERENCE_STEPS / (end_time - start_time + 1e-6)

    return (
        average_inference_time_sequential,
        average_fps_sequential,
        average_inference_time_split,
        average_fps__split,
    )


def benchmark(models_to_benchmark):
    results_dict = {}
    for model_name in models_to_benchmark:
        kwargs = {}
        if model_name in [
            "rt_detr",
            "optim_rt_detr",
            "deformable_detr",
            "optim_deformable_detr",
        ]:
            kwargs = {"disable_custom_kernels": True}
        model = (
            MODEL_NAMES_MODEL_CORRENSPONDENCE[model_name]
            .from_pretrained(MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name], **kwargs)
            .to("cuda")
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name]
        )
        print(f"Benchmarking {model_name}")
        print("Benchmarking eager model")
        (
            average_inference_time_sequential,
            average_fps_sequential,
            average_inference_time_split,
            average_fps__split,
        ) = benchmark_model(model, processor, model_name, "eager")
        print(
            f"Average inference time sequential: {average_inference_time_sequential:.3f} s"
        )
        print(f"Average FPS sequential: {average_fps_sequential:.3f}")
        print(f"Average inference time split: {average_inference_time_split:.3f} s")
        print(f"Average FPS split: {average_fps__split:.3f}")
        print("\n")

        results_dict[model_name] = {
            "eager": {
                "average_inference_time_sequential": average_inference_time_sequential,
                "average_fps_sequential": average_fps_sequential,
                "average_inference_time_split": average_inference_time_split,
                "average_fps_split": average_fps__split,
            }
        }

        print("benchmarking compiled model")
        model_compiled = torch.compile(model, mode="reduce-overhead")
        (
            average_inference_time_sequential,
            average_fps_sequential,
            average_inference_time_split,
            average_fps__split,
        ) = benchmark_model(model_compiled, processor, model_name, "compiled")
        print(
            f"Average inference time sequential: {average_inference_time_sequential:.3f} s"
        )
        print(f"Average FPS sequential: {average_fps_sequential:.3f}")
        print(f"Average inference time split: {average_inference_time_split:.3f} s")
        print(f"Average FPS split: {average_fps__split:.3f}")
        print("\n")

        results_dict[model_name]["compiled"] = {
            "average_inference_time_sequential": average_inference_time_sequential,
            "average_fps_sequential": average_fps_sequential,
            "average_inference_time_split": average_inference_time_split,
            "average_fps__split": average_fps__split,
        }

        if model_name in [
            "rt_detr",
            "optim_rt_detr",
            "deformable_detr",
            "optim_deformable_detr",
        ]:
            del model_compiled
            del model
            model = (
                MODEL_NAMES_MODEL_CORRENSPONDENCE[model_name]
                .from_pretrained(
                    MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name],
                    disable_custom_kernels=False,
                )
                .to("cuda")
            )
            print("benchmarking eager model with cuda kernel for deformable attention")
            (
                average_inference_time_sequential,
                average_fps_sequential,
                average_inference_time_split,
                average_fps__split,
            ) = benchmark_model(model, processor, model_name, "eager_custom_kernel")
            print(
                f"Average inference time sequential: {average_inference_time_sequential:.3f} s"
            )
            print(f"Average FPS sequential: {average_fps_sequential:.3f}")
            print(f"Average inference time split: {average_inference_time_split:.3f} s")
            print(f"Average FPS split: {average_fps__split:.3f}")
            print("\n")

            results_dict[model_name]["eager_custom_kernel"] = {
                "average_inference_time_sequential": average_inference_time_sequential,
                "average_fps_sequential": average_fps_sequential,
                "average_inference_time_split": average_inference_time_split,
                "average_fps__split": average_fps__split,
            }

            print(
                "benchmarking compiled model with cuda kernel for deformable attention"
            )
            model_compiled = torch.compile(model, mode="reduce-overhead")
            (
                average_inference_time_sequential,
                average_fps_sequential,
                average_inference_time_split,
                average_fps__split,
            ) = benchmark_model(
                model_compiled, processor, model_name, "compiled_custom_kernel"
            )
            print(
                f"Average inference time sequential: {average_inference_time_sequential:.3f} s"
            )
            print(f"Average FPS sequential: {average_fps_sequential:.3f}")
            print(f"Average inference time split: {average_inference_time_split:.3f} s")
            print(f"Average FPS split: {average_fps__split:.3f}")
            print("\n")

            results_dict[model_name]["compiled_custom_kernel"] = {
                "average_inference_time_sequential": average_inference_time_sequential,
                "average_fps_sequential": average_fps_sequential,
                "average_inference_time_split": average_inference_time_split,
                "average_fps__split": average_fps__split,
            }

        del model
        del model_compiled

    return results_dict


if __name__ == "__main__":
    results = benchmark(
        [
            "detr",
            "deformable_detr",
            "optim_deformable_detr",
            "rt_detr",
            "optim_rt_detr",
        ]
    )
    # dump to pretty json, limit to 2 decimal places
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4, default=lambda x: round(x, 2))

    print("Benchmarking done")
    print("Results saved to benchmark_results.json")
