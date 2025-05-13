import json, base64, io
from PIL import Image
import yaml
from model_handler import ModelHandler

def init_context(context):
    # Cargar checkpoint y pasar tamaño genérico; se sustituye en handler
    checkpoint = "/opt/nuclio/langrs/sam_vit_h_4b8939.pth"
    # placeholder; será recalculado en cada petición
    context.user_data.model = None
    context.user_data.checkpoint = checkpoint

def handler(context, event):
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")

    # Obtener input del usuario
    prompt = data.get("userTextInput") or False
    if not prompt:
        return context.Response(
            body=json.dumps({"error": "No prompt provided"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    print("prompt obtenido AUTO:", prompt)

    # Obtener label de la imagen
    label = data.get("cvatLabel") or False
    if not label:
        return context.Response(
            body=json.dumps({"error": "No label provided"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )
    print("label obtenido CVAT:", label)

    # Inicializar modelo con dimensiones reales
    image_size = (image.height, image.width)
    model = ModelHandler(context.user_data.checkpoint, image_size)
    context.user_data.model = model

    results = model.infer(image, prompt, label)
    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
