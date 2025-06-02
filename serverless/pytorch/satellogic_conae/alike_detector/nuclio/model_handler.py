import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import find_contours, approximate_polygon

MASK_THRESHOLD = 0.5

def to_cvat_mask(box, mask_2d):
    """
    Crop the full‐image mask to the box, flatten and
    append [xtl, ytl, xbr, ybr] for CVAT RLE.
    """
    xtl, ytl, xbr, ybr = map(int, box)
    crop = mask_2d[ytl : ybr + 1, xtl : xbr + 1]
    flat = crop.flatten().astype(int).tolist()
    flat.extend([xtl, ytl, xbr, ybr])
    return flat

class ModelHandler:
    def __init__(self, checkpoint: str, image_size: tuple):
        """
        checkpoint: path to sam2 checkpoint (.pt)
        image_size: (height, width) of expected input images
        """
        self.image_h, self.image_w = image_size
        # Build the core SAM2 model
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        # self.segmenter = Sam2Segmenter(checkpoint=checkpoint, device='cuda')

    def infer(self, image, prompt: str, label: str):
        img = np.array(image)
        # feed the image
        self.predictor.set_image(img)
        # text prompt support depends on how you wrap it;
        # here we pass a single‐element list with your prompt:
        masks, scores, boxes = self.predictor.predict([prompt])

        results = []
        for mask, score, box in zip(masks, scores, boxes):
            # convert to CVAT RLE
            cvat_mask = to_cvat_mask(box, mask)

            # extract one polygon (largest contour) if needed
            contours = find_contours(mask.astype(float), MASK_THRESHOLD)
            if contours:
                poly = approximate_polygon(
                    np.flip(contours[0], axis=1),
                    tolerance=2.5
                ).ravel().tolist()
            else:
                poly = []

            results.append({
                "label":      label,
                "confidence": f"{score:.4f}",
                "type":       "mask",
                "mask":       cvat_mask,
                "points":     poly,
                "attributes": []
            })

        return results

# from segment_anything import sam_model_registry, SamPredictor

# class ModelHandler:
#     def __init__(self, checkpoint, image_size):
#         self.checkpoint = checkpoint
#         self.image_h, self.image_w = image_size

#     def _iou(self, box1, box2):
#         """Intersection over Union of two boxes (x1,y1,x2,y2)."""
#         xa = max(box1[0], box2[0])
#         ya = max(box1[1], box2[1])
#         xb = min(box1[2], box2[2])
#         yb = min(box1[3], box2[3])
#         inter_w = max(0, xb - xa)
#         inter_h = max(0, yb - ya)
#         inter = inter_w * inter_h
#         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
#         union = area1 + area2 - inter
#         return inter / union if union else 0.0

#     def _filter_boxes_nms(self, boxes, thresh):
#         """
#         Greedy NMS to remove boxes with IoU > thresh.
#         """
#         # sort descending by area
#         boxes = sorted(boxes,
#                     key=lambda b: (b[2]-b[0])*(b[3]-b[1]),
#                     reverse=True)
#         keep = []
#         for b in boxes:
#             if all(self._iou(b, k) < thresh for k in keep):
#                 keep.append(b)
#         return keep

#     def infer(self, image, prompt, label):
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