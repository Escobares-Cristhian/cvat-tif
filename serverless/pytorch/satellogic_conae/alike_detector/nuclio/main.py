import json
import base64
import io
import os
from PIL import Image
import xml.etree.ElementTree as ET
from cvat_sdk.api_client import Configuration, ApiClient

from model_handler import ModelHandler
import torch
import gc

try:
    import cvat_config
    # Load CVAT connection settings from the cvat_config module
    CVAT_SCHEME = cvat_config.CVAT_SCHEME
    CVAT_HOST   = cvat_config.CVAT_HOST
    CVAT_PORT   = cvat_config.CVAT_PORT
    CVAT_USERNAME   = cvat_config.CVAT_USERNAME
    CVAT_PASSWORD   = cvat_config.CVAT_PASSWORD
except ImportError:
    print("cvat_config module not found. Ensure it is in the same directory as this script.")
    print("This file is not tracked by git, so it must be created manually with the form:")
    print("""
CVAT_SCHEME = "http"
CVAT_HOST   = <host as string: default is localhost or IP>
CVAT_PORT   = <port as string: default is 8080>
CVAT_USERNAME   = <username as string>
CVAT_PASSWORD   = <password as string>
""")
    print(1/0)


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

def get_annotations_from_cvat_annotations(task_id: int, frame_number: int, label_id: int):
    """
    Retrieves and filters mask annotations from CVAT for a given task, frame, and label.
    Raises ConnectionError if unable to reach the CVAT server.
    """
    global CVAT_SCHEME, CVAT_HOST, CVAT_PORT, CVAT_USERNAME, CVAT_PASSWORD

    if task_id is None:
        raise ValueError("task_id cannot be None")
    if frame_number is None:
        raise ValueError("frame_number cannot be None")

    # Construct base URL without trailing path; SDK appends /api internally
    base_url = f"{CVAT_SCHEME}://{CVAT_HOST}:{CVAT_PORT}"

    # Configure the low-level SDK client
    config = Configuration(
        host     = base_url,
        username = CVAT_USERNAME,
        password = CVAT_PASSWORD,
    )

    # Attempt to retrieve annotations
    try:
        with ApiClient(config) as api_client:
            parsed, _ = api_client.tasks_api.retrieve_annotations(
                id=int(task_id),
                _parse_response=True,
            )
    except Exception as e:
        raise ConnectionError(f"Failed to connect to CVAT at {base_url}/api: {e}")

    # DEBUG: print keys of a single shape in parsed.shapes
    sample = parsed.shapes[0] if parsed.shapes else None
    if sample:
        print("Finded annotations of any label:", len(parsed.shapes))

    # Filter by frame, shape type, and label
    annotations = [
        shape for shape in parsed.shapes
        if getattr(shape, "frame", None) == int(frame_number)
        and str(getattr(shape, "type", None)) == "mask"
        and getattr(shape, "label_id", None) == int(label_id)
    ]

    return annotations

def handler(context, event):
    init_context(context)
    print("\nhandler executed")
    data = event.body

    print("data['image'] type:", type(data["image"]))

    taskId = data["taskId"]
    frame = data["frame"]
    print(f"taskId: {taskId}, frame: {frame}")

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
    label_name = data.get("cvatLabel")
    if not label_name:
        return context.Response(
            body=json.dumps({"error": "No label provided"}),
            content_type="application/json",
            status_code=400,
        )

    # Get the CVAT label id
    label_id = data.get("cvatLabelId")
    if not label_id:
        return context.Response(
            body=json.dumps({"error": "No label provided"}),
            content_type="application/json",
            status_code=400,
        )

    print("label obtenido CVAT:", label_name, type(label_name))
    print("label id obtenido CVAT:", label_id, type(label_id))

    # Get annotations from CVAT annotations of label
    annotations = get_annotations_from_cvat_annotations(taskId, frame, label_id)
    print("annotations:")
    print(f"type: {type(annotations)}")
    print(f"len: {len(annotations)}")
    if len(annotations) > 0:
        print(f"first annotation type: {type(annotations[0])}")
        print(f"first annotation dir attributes: {dir(annotations[0])}")
        print(f"first annotation: {annotations[0]}")
    annotations_proc = []
    for annotation in annotations:
        annotations_proc.append({
            "id": annotation.id,
            "type": annotation.type,
            "points": annotation.points if hasattr(annotation, 'points') else None,  # Check if points exists
        })
    # print(f"shape: {annotation[0].mask.shape if annotation else 'No annotation found'}")

    # Instantiate and run inference
    image_size = (image.height, image.width)
    print("Full image size:", image_size)


    # Instantiate model once (and clear caches beforehand)
    if not hasattr(context.user_data, "model"):
        print("Loading SAM2 model for the first time…")
        # reuse your existing init_context to drop GPU & CPU caches
        init_context(context)
        context.user_data.model = ModelHandler(image_size)
    model = context.user_data.model
    print("Reusing existing model instance")

    # Run inference without building computation graphs
    with torch.no_grad():
        print("Running model.infer")
        results = model.infer(image, label_name, annotations_proc)

    # Return CVAT‐compatible JSON
    print("Returning results")
    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
