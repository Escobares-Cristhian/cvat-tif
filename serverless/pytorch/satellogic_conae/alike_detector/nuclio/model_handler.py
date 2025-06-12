import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from skimage.measure import find_contours, approximate_polygon
import torch
import math
# from pycocotools import mask as maskUtils

MASK_THRESHOLD = 0.5

def to_cvat_mask(box, mask_2d):
    """
    box: [x0, y0, w, h]   (SAM2 is XYWH!)
    mask_2d: full-image binary mask (0/1 numpy array)

    returns: a flat List[int] of length (h*w + 4), where
      - the first h*w entries are your mask bits in row-major
      - the last four are [x0, y0, x1, y1] for CVAT to splice off
    """
    # 1) unpack and convert XYWH -> XYXY
    xtl, ytl, w, h = map(int, box)
    xbr, ybr = xtl + w, ytl + h

    # Ensure numpy array and crop to the box region
    if not isinstance(mask_2d, np.ndarray):
        mask_2d = np.array(mask_2d)
    crop = mask_2d[ytl : ybr + 1, xtl : xbr + 1]

    # Flatten the mask pixels
    mask_flat = crop.flat[:].tolist()

    # Append the box coords at the very end (CVAT will read them from the tail)
    mask_flat.extend([xtl, ytl, xbr, ybr])

    return mask_flat

class ModelHandler:
    def __init__(self, image_size: tuple):
        """
        checkpoint: path to sam2 checkpoint (.pt)
        image_size: (height, width) of expected input images
        """
        self.image_h, self.image_w = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.mask_generator = SAM2AutomaticMaskGenerator(
            build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device, apply_postprocessing=False),
            # use_m2m=True,           # <-- enable mask-to-mask refinement
            # multimask_output=False, # <-- disable multimask output
            points_per_side=64,           # finer grid → better boundary detail
            pred_iou_thresh=0,            # no mask IoU filtering
            stability_score_thresh=0.8,   # stricter stability filtering
            box_nms_thresh=0.85,          # IoU threshold for NMS (for similar masks)

            # --- crop parameters ---
            crop_n_layers=1,                   # run one extra layer of crops
            crop_overlap_ratio=0.5,            # 50% overlap between tiles
            crop_n_points_downscale_factor=1,  # downscale points by 1x in crops
            crop_nms_thresh=0.85,               # IoU threshold for NMS in crops (for similar masks)


            # --- GPU parameters ---
            points_per_batch=64,  # number of points to process in parallel (default: 64)
            output_mode="binary_mask",  # output binary masks (default: "binary_mask" but consumes more memory, alternative: "coco_rle")

            # --- post‐processing ---
            min_mask_region_area=5,   # drop tiny objects <5 px

        )

    def _segments_to_cvat_masks(self, segments, label):
        results = []
        for seg in segments:
            # 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box
            mask  = seg["segmentation"]
            box   = seg["bbox"]
            score = seg.get("stability_score", 1.0)

            # convert to CVAT mask format
            cvat_mask = to_cvat_mask(box, mask)

            if cvat_mask is None:
                continue       # drop empty proposals

            # extract one polygon (largest contour) if needed
            contour = find_contours(mask, MASK_THRESHOLD)
            if len(contour) == 0:
                continue       # drop empty proposals
            contour = approximate_polygon(np.flip(contour[0], axis=1), tolerance=2.5)

            results.append({
                "label":      label,
                "confidence": f"{score:.4f}",
                "type":       "mask",
                "mask":       cvat_mask,
                "points":     contour.ravel().tolist(),
                "attributes": []
            })
        return results

    def infer(self, image, label: str):
        img = np.array(image)

        # Get all segmentations from the image
        segments = self.mask_generator.generate(img)
        print("Cantidad de segmentos obtenidos:", len(segments))

        # Post-process segments to CVAT mask format
        results = self._segments_to_cvat_masks(segments, label)
        print("Cantidad de segmentos procesados:", len(results))

        return results

# from segment_anything import sam_model_registry, SamPredictor

# class ModelHandler:
#     def __init__(self, checkpoint, image_size):
#         self.checkpoint = checkpoint
#         self.image_h, self.image_w = image_size

#     def infer(self, image, label):
#         # 1) Obtener las anotaciones de la imagen y convertirlas a embeddings
#         # Para "Detect Everything" se usa el predictor de SAM
#         # Para los embeddings se puede usar SAM o SAM2 (más rápido que SAM)

#         # --------------------------------------------------------------------------
#         # 0) Defino tamaño máximo de objeto indivisible "M"

#         # 1) Defino tamaño del lado de la ventana:
#             # 1.a) Lado Lx1024 con L el valor mínimo tal que L*1024 >= M
#             # 1.b) Lado máx(1024, M + a) con "a" un valr fijo (ó porcentaje) a definir

#         # 2) Defino overlap:
#             # 2.a) Overlap fijo (número de píxeles)
#             # 2.b) Overlap variable (porcentaje de la ventana)
#             # 2.c) Mixto: (min_overlap, b*ventana_size, max_overlap). Ej: (50, b*ventana_size) con b E (0, 0.5)

#         # 3) Método propuesto de identificación de objetos similares:
#             # 3.a) Filtrado de ventanas por embeddings:
#                 # 3.a.1) Se obtiene el embedding de cada ventana
#                 # 3.a.2) Se comparan los embeddings de cada ventana con los de las anotaciones
#                     # 3.a.3.a) Se devuelve las ventanas (Poligon)
#                     # 3.a.3.b) Se segmenta con "SAM detect everything" y se comparan los embeddings de las segmentaciones con los de las anotaciones (Mask)
#             # 3.b) Sin filtrado de ventanas por embeddings:
#                 # 3.b.1) En cada ventana se corre "SAM detect everything" con tamaño b de grilla para obtener las segmentaciones
#                 # 3.b.2) Se comparan los embeddings de cada segmentación con los de las anotaciones
#                 # 3.b.3) Se devuelve las segmentaciones (Mask)

#         # 4) Unión de máscaras de diferentes ventanas (caso Mask):
#             # 4.1) Si no se superponen bbox -> No se hace nada
#             # 4.2) Si se superponen bbox, pero píxel a píxel no se superponen -> No se hace nada
#             # 4.3) Si se superponen bbox y se superponen en algún píxel:
#                 # 4.3.1) Si son del mismo label -> Se unen
#                 # 4.3.2) Si son de diferentes labels -> No se hace nada
#             # De esta forma se unen únicamente los objetos con píxeles superpuestos del mismo objeto
#             # caso contrario, se mantienen las máscaras separadas

#         # --------------------------------------------------------------------------