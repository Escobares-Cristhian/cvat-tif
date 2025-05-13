import os, shutil, math, cv2
import numpy as np
from langrs import LangRS as _OrigLangRS
from skimage.measure import find_contours, approximate_polygon

import matplotlib.pyplot as plt

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
    # Unpack and cast to ints
    xtl, ytl, xbr, ybr = map(int, box)

    # Ensure numpy array and crop to the box region
    if not isinstance(mask_2d, np.ndarray):
        mask_2d = np.array(mask_2d)
    crop = mask_2d[ytl : ybr + 1, xtl : xbr + 1]

    # Flatten the mask pixels
    mask_flat = crop.flat[:].tolist()

    # Append the box coords at the very end (CVAT will read them from the tail)
    mask_flat.extend([xtl, ytl, xbr, ybr])

    return mask_flat

def segm_postprocess(box: list, raw_cls_mask, im_h, im_w):
    # Unpack and cast to ints
    xmin, ymin, xmax, ymax = map(int, box)

    width = xmax - xmin + 1
    height = ymax - ymin + 1

    result = np.zeros((im_h, im_w), dtype=np.uint8)
    resized_mask = cv2.resize(raw_cls_mask, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    # extract the ROI of the image
    result[ymin:ymax + 1, xmin:xmax + 1] = (resized_mask > MASK_THRESHOLD).astype(np.uint8) * 255

    return result


class ModelHandler:
    def __init__(self, checkpoint, image_size):
        self.checkpoint = checkpoint
        self.image_h, self.image_w = image_size

    def infer(self, image, prompt):
        # 3) Generar y filtrar cajas
        model = LangRS(np.array(image), prompt, output_path="/tmp", checkpoint=self.checkpoint)
        boxes = model.generate_boxes(window_size=1000, overlap=200,
                                     box_threshold=MASK_THRESHOLD, text_threshold=MASK_THRESHOLD)
        filtered_boxes = model.outlier_rejection().get("zscore", boxes)

        # 4) Generar máscaras *planas* (raw)
        raw_masks = model.generate_masks(boxes=filtered_boxes, window_size=1024, overlap=200)

        # 5) Generar máscaras *reales* (segm)
        results = []
        for box in filtered_boxes:
            # a) Codificar a RLE para CVAT
            cvat_mask = to_cvat_mask(box, raw_masks)  # requiere máscara 2-D :contentReference[oaicite:7]{index=7}
            # b) (Opcional) extraer contornos para polígonos
            contour = find_contours(raw_masks, MASK_THRESHOLD)
            contour = approximate_polygon(np.flip(contour[0], axis=1), tolerance=2.5)
            results.append({
                "label":      prompt,
                "confidence": str(1.0),
                "type":       "mask",       # tipo “mask” :contentReference[oaicite:8]{index=8}
                "mask":       cvat_mask,    # RLE + [xtl, ytl, xbr, ybr]
                "points":     contour.ravel().tolist(),
                "attributes": []            # evita undefined.reduce :contentReference[oaicite:9]{index=9}
            })

        return results
