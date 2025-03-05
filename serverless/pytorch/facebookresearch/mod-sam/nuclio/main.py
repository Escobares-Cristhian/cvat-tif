"""
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""

import json
import base64
import io
from PIL import Image
from model_handler import ModelHandler

def init_context(context):
    context.logger.info("Init context... 0%")
    model = ModelHandler()
    context.user_data.model = model
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("call handler")
    data = event.body
    # Decode the image from base64.
    buf = io.BytesIO(base64.b64decode(data["image"]))
    image = Image.open(buf).convert("RGB")
    # Get selected points (if any)
    positive_points = data.get("positive_points", [])
    negative_points = data.get("negative_points", [])
    
    # Process the image with our modified ModelHandler.
    features, crop_bbox = context.user_data.model.handle(image, positive_points, negative_points)

    return context.Response(
        body=json.dumps({
            'blob': base64.b64encode(
                features.cpu().numpy() if features.is_cuda else features.numpy()
            ).decode(),
            'crop_bbox': crop_bbox,
        }),
        headers={},
        content_type='application/json',
        status_code=200
    )

