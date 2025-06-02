import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler

# Point to the SAM 2 checkpoint that we bake into the image
SAM2_CHECKPOINT = "/opt/nuclio/sam2/sam2_hiera_large.pt"

def init_context(context):
    # Store the checkpoint path for use in each invocation
    context.user_data.checkpoint = SAM2_CHECKPOINT

def handler(context, event):
    data = event.body

    # Decode the base64 image
    image = Image.open(io.BytesIO(base64.b64decode(data["image"]))).convert("RGB")

    # Get the text prompt
    prompt = data.get("userTextInput")
    if not prompt:
        return context.Response(
            body=json.dumps({"error": "No prompt provided"}),
            content_type="application/json",
            status_code=400,
        )

    # Get the CVAT label
    label = data.get("cvatLabel")
    if not label:
        return context.Response(
            body=json.dumps({"error": "No label provided"}),
            content_type="application/json",
            status_code=400,
        )

    # Instantiate and run inference
    image_size = (image.height, image.width)
    model = ModelHandler(context.user_data.checkpoint, image_size)
    results = model.infer(image, prompt, label)

    # Return CVAT‐compatible JSON
    return context.Response(
        body=json.dumps(results),
        content_type="application/json",
        status_code=200,
    )
