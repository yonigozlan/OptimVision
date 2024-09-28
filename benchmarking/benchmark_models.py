import json
import time

import requests
import torch

# from optim_deformable_detr import OptimDeformableDetrForObjectDetection
# from optim_rt_detr import OptimRTDetrForObjectDetection
from PIL import Image

from transformers import (
    AutoProcessor,
    DetrForObjectDetection,
)
from transformers import (
    DeformableDetrForObjectDetection as OptimDeformableDetrForObjectDetection,
)

# from transformers import (
#     OmDetTurboForObjectDetection as OptimOmDetTurboForObjectDetection,
# )
from transformers import (
    RTDetrForObjectDetection as OptimRTDetrForObjectDetection,
)
from transformers.models.deformable_detr.modeling_deformable_detr_baseline import (
    DeformableDetrForObjectDetection,
)

# from transformers.models.omdet_turbo.modeling_omdet_turbo_baseline import (
#     OmDetTurboForObjectDetection,
# )

# from transformers.models.rt_detr.modeling_rt_detr_baseline import (
#     RTDetrForObjectDetection,
# )
IMAGE_SIZE = (640, 640)
MANUAL_INFERENCE_STEPS = 1000
LOG_DIR = "./log_benchmark_deformable_detr_two_stage"
OUTPUT_FILE_PATH = "./benchmark_results_deformable_detr_two_stage.json"

MODEL_NAMES_MODEL_CORRESPONDENCE = {
    # "rt_detr": RTDetrForObjectDetection,
    "optim_rt_detr": OptimRTDetrForObjectDetection,
    "deformable_detr": DeformableDetrForObjectDetection,
    "optim_deformable_detr": OptimDeformableDetrForObjectDetection,
    "detr": DetrForObjectDetection,
    # "omdet_turbo": OmDetTurboForObjectDetection,
    # "optim_omdet_turbo": OptimOmDetTurboForObjectDetection,
}


MODEL_NAMES_WEIGHTS_CORRESPONDENCE = {
    "rt_detr": "PekingU/rtdetr_r101vd",
    "optim_rt_detr": "PekingU/rtdetr_r101vd",
    "deformable_detr": "SenseTime/deformable-detr",
    "optim_deformable_detr": "SenseTime/deformable-detr",
    "detr": "facebook/detr-resnet-50",
    "omdet_turbo": "../omdet-turbo-tiny-timm",
    "optim_omdet_turbo": "../omdet-turbo-tiny-timm",
}


def get_sample_input_image():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


def benchmark_model(model, processor, model_name, experiment, dtype=torch.float32):
    image = get_sample_input_image()
    inputs = processor(
        images=image,
        # text=["person", "ball", "shoe"],
        return_tensors="pt",
        # size=IMAGE_SIZE,
    ).to("cuda", dtype=dtype)

    outputs = None
    results = None
    # pytorch/tensorboard profiling
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=2, warmup=3, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"./{LOG_DIR}/{model_name}",
            worker_name=f"{experiment}",
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        with torch.no_grad():
            for _ in range(10):
                profiler.step()
                outputs = model(**inputs)
                outputs[0].cpu()
    del outputs, results
    outputs = None
    results = None

    # manual profiling
    start_time = time.time()
    with torch.no_grad():
        for _ in range(MANUAL_INFERENCE_STEPS):
            outputs = model(**inputs)
            outputs[0].cpu()

    end_time = time.time()
    average_inference_time = (end_time - start_time) / MANUAL_INFERENCE_STEPS
    average_fps = MANUAL_INFERENCE_STEPS / (end_time - start_time + 1e-6)

    del outputs, results

    return (
        average_inference_time,
        average_fps,
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
            MODEL_NAMES_MODEL_CORRESPONDENCE[model_name]
            .from_pretrained(MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name], **kwargs)
            .to("cuda")
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name]
        )
        print(f"Benchmarking {model_name}")
        print("Benchmarking eager model (fp32)")
        (
            average_inference_time,
            average_fps,
        ) = benchmark_model(model, processor, model_name, "eager")
        print(f"Average inference time: {average_inference_time:.3f} s")
        print(f"Average FPS: {average_fps:.3f}")
        print("\n")

        results_dict[model_name] = {
            "eager": {
                "average_inference_time": average_inference_time,
                "average_fps": average_fps,
            }
        }

        print("benchmarking compiled model")
        model_compiled = torch.compile(model, mode="reduce-overhead")
        (
            average_inference_time,
            average_fps,
        ) = benchmark_model(model_compiled, processor, model_name, "compiled")
        print(f"Average inference time: {average_inference_time:.3f} s")
        print(f"Average FPS: {average_fps:.3f}")
        print("\n")

        results_dict[model_name]["compiled"] = {
            "average_inference_time": average_inference_time,
            "average_fps": average_fps,
        }
        del model_compiled

        print("Benchmarking eager model (fp16)")
        model = model.to(dtype=torch.float16)
        (
            average_inference_time,
            average_fps,
        ) = benchmark_model(
            model, processor, model_name, "eager_fp16", dtype=torch.float16
        )
        print(f"Average inference time: {average_inference_time:.3f} s")
        print(f"Average FPS: {average_fps:.3f}")
        print("\n")

        results_dict[model_name]["eager_fp16"] = {
            "average_inference_time": average_inference_time,
            "average_fps": average_fps,
        }

        del model

        print("benchmarking compiled model (fp16)")
        model = (
            MODEL_NAMES_MODEL_CORRESPONDENCE[model_name]
            .from_pretrained(MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name], **kwargs)
            .to("cuda")
            .to(dtype=torch.float16)
        )
        model_compiled = torch.compile(model, mode="reduce-overhead")
        (
            average_inference_time,
            average_fps,
        ) = benchmark_model(
            model_compiled, processor, model_name, "compiled_fp16", dtype=torch.float16
        )
        print(f"Average inference time: {average_inference_time:.3f} s")
        print(f"Average FPS: {average_fps:.3f}")
        print("\n")

        results_dict[model_name]["compiled_fp16"] = {
            "average_inference_time": average_inference_time,
            "average_fps": average_fps,
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
                MODEL_NAMES_MODEL_CORRESPONDENCE[model_name]
                .from_pretrained(
                    MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name],
                    disable_custom_kernels=False,
                )
                .to("cuda")
            )
            print("benchmarking eager model with cuda kernel for deformable attention")
            (
                average_inference_time,
                average_fps,
            ) = benchmark_model(model, processor, model_name, "eager_custom_kernel")
            print(f"Average inference time: {average_inference_time:.3f} s")
            print(f"Average FPS: {average_fps:.3f}")
            print("\n")

            results_dict[model_name]["eager_custom_kernel"] = {
                "average_inference_time": average_inference_time,
                "average_fps": average_fps,
            }

            print(
                "benchmarking compiled model with cuda kernel for deformable attention"
            )
            model_compiled = torch.compile(model, mode="reduce-overhead")
            (
                average_inference_time,
                average_fps,
            ) = benchmark_model(
                model_compiled, processor, model_name, "compiled_custom_kernel"
            )
            print(f"Average inference time: {average_inference_time:.3f} s")
            print(f"Average FPS: {average_fps:.3f}")
            print("\n")

            results_dict[model_name]["compiled_custom_kernel"] = {
                "average_inference_time": average_inference_time,
                "average_fps": average_fps,
            }
            del model_compiled

            print(
                "benchmarking eager model with cuda kernel for deformable attention (fp16)"
            )
            model = model.to(dtype=torch.float16)
            (
                average_inference_time,
                average_fps,
            ) = benchmark_model(
                model,
                processor,
                model_name,
                "eager_custom_kernel_fp16",
                dtype=torch.float16,
            )
            print(f"Average inference time: {average_inference_time:.3f} s")
            print(f"Average FPS: {average_fps:.3f}")
            print("\n")

            results_dict[model_name]["eager_custom_kernel_fp16"] = {
                "average_inference_time": average_inference_time,
                "average_fps": average_fps,
            }

            del model

            print(
                "benchmarking compiled model with cuda kernel for deformable attention (fp16)"
            )
            model = (
                MODEL_NAMES_MODEL_CORRESPONDENCE[model_name]
                .from_pretrained(
                    MODEL_NAMES_WEIGHTS_CORRESPONDENCE[model_name],
                    disable_custom_kernels=False,
                )
                .to("cuda")
                .to(dtype=torch.float16)
            )
            model_compiled = torch.compile(model, mode="reduce-overhead")
            (
                average_inference_time,
                average_fps,
            ) = benchmark_model(
                model_compiled,
                processor,
                model_name,
                "compiled_custom_kernel_fp16",
                dtype=torch.float16,
            )
            print(f"Average inference time: {average_inference_time:.3f} s")
            print(f"Average FPS: {average_fps:.3f}")
            print("\n")

            results_dict[model_name]["compiled_custom_kernel_fp16"] = {
                "average_inference_time": average_inference_time,
                "average_fps": average_fps,
            }

        del model
        del model_compiled

    return results_dict


if __name__ == "__main__":
    results = benchmark(
        [
            # "detr",
            "deformable_detr",
            "optim_deformable_detr",
            # "rt_detr",
            # "optim_rt_detr",
            # "omdet_turbo",
            # "optim_omdet_turbo",
        ]
    )
    # dump to pretty json, limit to 2 decimal places
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f, indent=4, default=lambda x: round(x, 2))

    print("Benchmarking done")
    print(f"Results saved to {OUTPUT_FILE_PATH}")
