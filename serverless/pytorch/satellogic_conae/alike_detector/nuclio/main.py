import json
import base64
import io
import os
from PIL import Image
from model_handler import ModelHandler
import torch
import gc
from cvat_sdk.api_client import Configuration, ApiClient

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

def get_embeddings_from_cvat_annotations(data, YOUR_LABEL):
    # 2. Extract the task ID and frame number that CVAT passed in
    task_id = data.get("task") or data.get("task_id")
    frame_number = data.get("frame") or data.get("frame_number")

    if task_id is None or frame_number is None:
        raise KeyError(f"Missing task_id/frame_number in payload; got keys {list(data)}")

    print(f"task_id: {task_id}, frame_number: {frame_number}")

    # 3. Configure the low-level CVAT SDK client (auto-read URL/creds from env)
    config = Configuration(
        host     = os.getenv("CVAT_HOST") + "/api",
        username = os.getenv("CVAT_USERNAME"),
        password = os.getenv("CVAT_PASSWORD"),
    )

    # 4. Fetch exactly those annotations for this frame
    with ApiClient(config) as api_client:
        parsed, _ = api_client.tasks_api.retrieve_annotations(
            id      = task_id,
            format_ = "CVAT 1.1",           # low-level export format :contentReference[oaicite:1]{index=1}
            frame   = frame_number,
            _parse_response = True,
        )

    # 5. Now `parsed.shapes` holds all shapes/masks; filter by your label:
    embeddings = []
    for shape in parsed.shapes:
        if shape.shape_type == "mask" and shape.label_name == YOUR_LABEL:
            # here you can feed shape.mask (RLE) back into your SAM2 predictor
            embeddings.append(shape)
    return embeddings

def handler(context, event):
    print("\nhandler executed")
    data = event.body

    print("data['image'] type:", type(data["image"]))

    print(f"task: {data.get('taskId')}, frame: {data.get('frame')}")

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

    # Get embeddings from CVAT annotations of label
    embeddings = get_embeddings_from_cvat_annotations(data, label)
    print("embeddings:")
    print(f"type: {type(embeddings)}")
    print(f"len: {len(embeddings)}")
    print(f"shape: {embeddings[0].mask.shape if embeddings else 'No embeddings found'}")


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
