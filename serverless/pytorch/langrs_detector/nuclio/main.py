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
    print("event:")
    print(f"\ttype: {type(event)}")
    print("keys:", list(vars(event).keys()))
    # Print all attribute values except "body"
    for key in vars(event).keys():
        if key != "body":
            print(f"\t{key}: {getattr(event, key)}")

    print("context:")
    print(f"\ttype: {type(context)}")
    keys = list(vars(context).keys())
    print("keys:", keys)
    for key in keys:
        print(f"\t{key}: {getattr(context, key)}")

    print("data:")
    print(f"\ttype: {type(data)}")
    print("keys:", list(data.keys()))
    # for key in data.keys():
    #     print(f"\t{key}: {data[key]}")

    prompt = data.get("userTextInput") or False
    if not prompt:
        return context.Response(
            body=json.dumps({"error": "No prompt provided"}),
            headers={},
            content_type="application/json",
            status_code=400,
        )

    print("prompt obtenido AUTO:", prompt)

    # Inicializar modelo con dimensiones reales
    image_size = (image.height, image.width)
    model = ModelHandler(context.user_data.checkpoint, image_size)
    context.user_data.model = model

    results = model.infer(image, prompt)
    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
