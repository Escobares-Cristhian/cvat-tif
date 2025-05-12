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

# 3) Función de post-proceso de máscara
# def segm_postprocess(box, raw_mask, im_h, im_w):
#     # Unpack
#     raw_xmin, raw_ymin, raw_xmax, raw_ymax = box

#     # Cast to Python ints (round or floor as you prefer)
#     xmin = int(raw_xmin.item() if hasattr(raw_xmin, "item") else raw_xmin)
#     ymin = int(raw_ymin.item() if hasattr(raw_ymin, "item") else raw_ymin)
#     xmax = int(raw_xmax.item() if hasattr(raw_xmax, "item") else raw_xmax)
#     ymax = int(raw_ymax.item() if hasattr(raw_ymax, "item") else raw_ymax)

#     # Also cast image dims
#     im_h = int(im_h)
#     im_w = int(im_w)

#     # Compute tile size
#     w = xmax - xmin + 1
#     h = ymax - ymin + 1

#     # Prepare full mask
#     full = np.zeros((im_h, im_w), dtype=np.uint8)

#     # Resize with valid int tuple
#     resized = cv2.resize(
#         raw_mask,
#         (w, h),
#         interpolation=cv2.INTER_CUBIC,
#     )

#     # Threshold & paste
#     mask_bin = (resized > MASK_THRESHOLD).astype(np.uint8) * 255

#     full[ymin : ymax + 1, xmin : xmax + 1] = mask_bin

#     return full

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

        print("\n\n")
        print(f"DEBUG: scanning raw_masks")
        print(f"\ttype(raw_masks): {type(raw_masks)}")
        print(f"\tlen(raw_masks): {len(raw_masks)}")
        print(f"\traw_masks.shape: {raw_masks.shape}")

        # # 4) DEBUG: devuelvo boxes como tipo rectangle
        # results = []
        # for box in filtered_boxes:
        #     xmin, ymin, xmax, ymax = map(int, box)
        #     results.append({
        #         "label":      prompt,
        #         "confidence": str(1.0),
        #         "type":       "rectangle",   # tipo “rectangle” :contentReference[oaicite:5]{index=5}
        #         "points":     [xmin, ymin, xmax, ymax],  # [xtl, ytl, xbr, ybr]
        #         "attributes": []             # evita undefined.reduce :contentReference[oaicite:9]{index=9}
        #     })

        results = []
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

        # results = []
        # # 5) Emparejar con zip() para evitar unpack error
        # for raw_mask, box in zip(raw_masks, filtered_boxes):
        #     # a) Reconstruir la máscara completa 2-D
        #     processed = segm_postprocess(box, raw_mask, self.image_h, self.image_w)
        #     # b) Codificar a RLE para CVAT
        #     cvat_mask = to_cvat_mask(box, processed)  # requiere máscara 2-D :contentReference[oaicite:7]{index=7}
        #     # c) (Opcional) extraer contornos para polígonos
        #     contours = find_contours(processed, MASK_THRESHOLD)
        #     contour = approximate_polygon(np.flip(contours[0], axis=1), tolerance=2.5)

        #     if len(contour) < 3:
        #         continue

        #     results.append({
        #         "label":      prompt,
        #         "confidence": str(1.0),
        #         "type":       "mask",       # tipo “mask” :contentReference[oaicite:8]{index=8}
        #         "mask":       cvat_mask,    # RLE + [xtl, ytl, xbr, ybr]
        #         "points":     contour.ravel().tolist(),
        #         "attributes": []            # evita undefined.reduce :contentReference[oaicite:9]{index=9}
        #     })


        print("-"*80)
        print("-"*80)
        print("-"*80)
        print("-"*80)
        print("Done!")
        return results
