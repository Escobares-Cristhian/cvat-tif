import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler
import torch
import gc

def init_context(context):
    # Release cache in RAM
    gc.collect()

    # Release cache in GPU memory
    if torch.cuda.is_available():
        gpu_gb_before = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU memory allocated before clearing: {gpu_gb_before:.2f} GB")

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        gpu_gb_after = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU memory allocated after clearing: {gpu_gb_after:.2f} GB")
        print(f"GPU memory cleared: {gpu_gb_before - gpu_gb_after:.2f} GB")
    print("Context initialized and caches cleared.")

    return

def handler(context, event):
    print("\nhandler executed")
    data = event.body

    print("data['image'] type:", type(data["image"]))

    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")

    # # Get the text prompt
    # prompt = data.get("userTextInput")
    # if not prompt:
    #     return context.Response(
    #         body=json.dumps({"error": "No prompt provided"}),
    #         content_type="application/json",
    #         status_code=400,
    #     )

    # print("prompt obtenido AUTO:", prompt)

    # Get the CVAT label
    label = data.get("cvatLabel")
    if not label:
        return context.Response(
            body=json.dumps({"error": "No label provided"}),
            content_type="application/json",
            status_code=400,
        )

    print("label obtenido CVAT:", label)

    # Instantiate and run inference
    image_size = (image.height, image.width)
    print("Full image size:", image_size)
    model = ModelHandler(image_size)
    context.user_data.model = model

    print("Trying model.infer")
    results = model.infer(image, label)

    # Return CVAT‐compatible JSON
    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
