import os, shutil, math, cv2
import numpy as np
from langrs import LangRS as _OrigLangRS
from skimage.measure import find_contours, approximate_polygon

MASK_THRESHOLD = 0.5

# 1) Monkey-patch para LangRS
class LangRS(_OrigLangRS):
    def __init__(self, image, prompt, output_path, checkpoint=None):
        if checkpoint:
            cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
            os.makedirs(cache_dir, exist_ok=True)
            dst = os.path.join(cache_dir, os.path.basename(checkpoint))
            if not os.path.exists(dst):
                shutil.copyfile(checkpoint, dst)
        super().__init__(image, prompt, output_path)

# 2) Función de codificación a RLE para CVAT
def to_cvat_mask(box, mask_2d):
    """
    Convert the cropped mask into CVAT’s flattened RLE‐style JSON array.

    Modified from:
        # Copyright (C) 2022 CVAT.ai Corporation
        # SPDX-License-Identifier: MIT
    """
    # Unpack box
    xtl, ytl, xbr, ybr = box

    # If any coordinate is a tensor or float, convert to Python int
    def to_int(val, name):
        if hasattr(val, "item"):
            v = val.item()
        else:
            v = val
        v_int = int(v)
        return v_int

    xtl_i = to_int(xtl, "xtl")
    ytl_i = to_int(ytl, "ytl")
    xbr_i = to_int(xbr, "xbr")
    ybr_i = to_int(ybr, "ybr")

    # Now slice – mask_2d must be a numpy array
    if not isinstance(mask_2d, np.ndarray):
        mask_2d = np.array(mask_2d)
    crop = mask_2d[ytl_i : ybr_i + 1, xtl_i : xbr_i + 1]

    # Flatten and return
    flattened = crop.flat[:].tolist()

    # Append the box coordinates
    flattened.extend([xtl_i, ytl_i, xbr_i, ybr_i])

    return flattened


# 3) Función de post-proceso de máscara
def segm_postprocess(box, raw_mask, im_h, im_w):
    # Unpack
    raw_xmin, raw_ymin, raw_xmax, raw_ymax = box

    # Cast to Python ints (round or floor as you prefer)
    xmin = int(raw_xmin.item() if hasattr(raw_xmin, "item") else raw_xmin)
    ymin = int(raw_ymin.item() if hasattr(raw_ymin, "item") else raw_ymin)
    xmax = int(raw_xmax.item() if hasattr(raw_xmax, "item") else raw_xmax)
    ymax = int(raw_ymax.item() if hasattr(raw_ymax, "item") else raw_ymax)

    # Also cast image dims
    im_h = int(im_h)
    im_w = int(im_w)

    # Compute tile size
    w = xmax - xmin + 1
    h = ymax - ymin + 1

    # Prepare full mask
    full = np.zeros((im_h, im_w), dtype=np.uint8)

    # Resize with valid int tuple
    resized = cv2.resize(
        raw_mask,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    # Threshold & paste
    mask_bin = (resized > MASK_THRESHOLD).astype(np.uint8) * 255

    full[ymin : ymax + 1, xmin : xmax + 1] = mask_bin

    return full


class ModelHandler:
    def __init__(self, checkpoint, image_size):
        self.checkpoint = checkpoint
        self.image_h, self.image_w = image_size

    def infer(self, image, prompt):
        # 3) Generar y filtrar cajas
        model = LangRS(np.array(image), prompt, output_path="/tmp", checkpoint=self.checkpoint)
        boxes = model.generate_boxes(window_size=1000, overlap=200,
                                     box_threshold=0.25, text_threshold=0.25)
        filtered_boxes = model.outlier_rejection().get("zscore", boxes)

        # 4) Generar máscaras *planas* (raw)
        raw_masks = model.generate_masks(boxes=filtered_boxes)  # lista de arrays 1-D :contentReference[oaicite:6]{index=6}

        results = []
        # 5) Emparejar con zip() para evitar unpack error
        for raw_mask, box in zip(raw_masks, filtered_boxes):
            # a) Reconstruir la máscara completa 2-D
            processed = segm_postprocess(box, raw_mask, self.image_h, self.image_w)
            # b) Codificar a RLE para CVAT
            cvat_mask = to_cvat_mask(box, processed)  # requiere máscara 2-D :contentReference[oaicite:7]{index=7}
            # c) (Opcional) extraer contornos para polígonos
            contours = find_contours(processed, MASK_THRESHOLD)
            contour = approximate_polygon(np.flip(contours[0], axis=1), tolerance=2.5)
            if len(contour) < 3:
                continue
            results.append({
                "label":      prompt,
                "confidence": str(1.0),
                "type":       "mask",       # tipo “mask” :contentReference[oaicite:8]{index=8}
                "mask":       cvat_mask,    # RLE + [xtl, ytl, xbr, ybr]
                "points":     contour.ravel().tolist(),
                "attributes": []            # evita undefined.reduce :contentReference[oaicite:9]{index=9}
            })

        print("Done!")
        return results
