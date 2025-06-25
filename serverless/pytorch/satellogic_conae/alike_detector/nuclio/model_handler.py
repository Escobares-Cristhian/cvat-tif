import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import find_contours, approximate_polygon
import torch
import types
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

def to_mask2d(cvat_mask):
    """
    Inverse of to_cvat_mask.

    Parameters
    ----------
    cvat_mask : sequence of int
        Flat list whose last four entries are [xtl, ytl, xbr, ybr]
        and whose preceding entries are mask bits in row-major order.

    Returns
    -------
    box : list of int
        [xtl, ytl, w, h] with w = xbr - xtl, h = ybr - ytl
    mask_2d : ndarray of shape (h, w), dtype uint8
        Reconstructed 2D binary mask (0/1).
    """
    # ensure we have a mutable list of ints
    data = list(map(int, cvat_mask))
    # extract box coords
    xtl, ytl, xbr, ybr = data[-4:]
    w = xbr - xtl
    h = ybr - ytl

    # the rest are the mask bits
    flat_mask = data[:-4]
    if len(flat_mask) != h * w:
        raise ValueError(f"Expected {h*w} mask bits, got {len(flat_mask)}")

    # reshape back to 2D
    mask_2d = np.array(flat_mask, dtype=np.uint8).reshape((h, w))

    # return box in XYWH form plus the mask
    return [xtl, ytl, w, h], mask_2d


# embeddings_proc = []
# for embedding in embeddings:
#     embeddings_proc.append({
#         "id": embedding.id,
#         "type": embedding.points,
#         "points": embedding.points if hasattr(embedding, 'points') else None,  # Check if points exists
#     })

class ModelHandler:
    def __init__(self, image_size: tuple):
        """
        checkpoint:
            Path to sam2 checkpoint (.pt)
        image_size:
            (height, width) of expected input images
        """
        self.image_h, self.image_w = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        # Instantiate the automatic mask generator
        self.mask_generator = SAM2AutomaticMaskGenerator(
            build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device, apply_postprocessing=False),
            # use_m2m=True,           # <-- enable mask-to-mask refinement
            # multimask_output=False, # <-- disable multimask output
            points_per_side=64,           # finer grid → better boundary detail
            pred_iou_thresh=0.5,            # no mask IoU filtering
            stability_score_thresh=0.5,   # stricter stability filtering
            box_nms_thresh=0.9,          # IoU threshold for NMS (for similar masks)

            # --- crop parameters ---
            crop_n_layers=1,                   # run one extra layer of crops
            crop_overlap_ratio=0.5,            # 50% overlap between tiles
            crop_n_points_downscale_factor=1,  # downscale points by 1x in crops
            crop_nms_thresh=0.9,               # IoU threshold for NMS in crops (for similar masks)

            # --- GPU parameters ---
            points_per_batch=64,  # number of points to process in parallel (default: 64)
            output_mode="binary_mask",  # output binary masks (default: "binary_mask" but consumes more memory, alternative: "coco_rle")

            # --- post‐processing ---
            min_mask_region_area=0,   # drop tiny objects <5 px
        )
        # self.mask_generator = SAM2AutomaticMaskGenerator(
        #     build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device, apply_postprocessing=False),
        #     # use_m2m=True,           # <-- enable mask-to-mask refinement
        #     # multimask_output=False, # <-- disable multimask output
        #     points_per_side=64,           # finer grid → better boundary detail
        #     pred_iou_thresh=0,            # no mask IoU filtering
        #     stability_score_thresh=0,   # stricter stability filtering
        #     box_nms_thresh=1.1,          # IoU threshold for NMS (for similar masks)

        #     # --- crop parameters ---
        #     crop_n_layers=1,                   # run one extra layer of crops
        #     crop_overlap_ratio=0.5,            # 50% overlap between tiles
        #     crop_n_points_downscale_factor=1,  # downscale points by 1x in crops
        #     crop_nms_thresh=1.1,               # IoU threshold for NMS in crops (for similar masks)

        #     # --- GPU parameters ---
        #     points_per_batch=64,  # number of points to process in parallel (default: 64)
        #     output_mode="binary_mask",  # output binary masks (default: "binary_mask" but consumes more memory, alternative: "coco_rle")

        #     # --- post‐processing ---
        #     min_mask_region_area=0,   # drop tiny objects <5 px
        # )

        # Patch reset_predictor to save the full-image embedding
        orig_reset = self.mask_generator.predictor.reset_predictor
        self.mask_generator.predictor.saved_image_embeddings = None
        def patched_reset(self_predictor):
            feats = getattr(self_predictor, "_features", None) or {}
            if "image_embed" in feats:
                self_predictor.saved_image_embeddings = feats["image_embed"]
            else:
                print(f"[patch] no 'image_embed' in features: {list(feats.keys())}")
            orig_reset()
        self.mask_generator.predictor.reset_predictor = types.MethodType(
            patched_reset,
            self.mask_generator.predictor
        )

        # Patch predict_masks to collect mask embeddings
        self.mask_embeddings = []
        decoder = self.mask_generator.predictor.model.sam_mask_decoder
        orig_predict = decoder.predict_masks
        def patched_predict(decoder_self,
                            image_embeddings, image_pe,
                            sparse_prompt_embeddings,
                            dense_prompt_embeddings,
                            repeat_image,
                            high_res_features=None):
            masks, iou_preds, tokens, obj_scores = orig_predict(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                repeat_image=repeat_image,
                high_res_features=high_res_features,
            )
            self.mask_embeddings.extend(tokens.detach().cpu().tolist())
            return masks, iou_preds, tokens, obj_scores
        decoder.predict_masks = types.MethodType(patched_predict, decoder)

    def _segments_to_cvat_masks(self, segments, label):
        results = []
        valid_indices = []
        count = 0
        for idx, seg in enumerate(segments):
            mask = seg["segmentation"]
            box = seg["bbox"]
            score = seg.get("stability_score", 1.0)
            cvat = to_cvat_mask(box, mask)
            contour = find_contours(mask, MASK_THRESHOLD)
            if not contour:
                count += 1
                continue
            valid_indices.append(idx)
            poly = approximate_polygon(np.flip(contour[0], axis=1), tolerance=2.5)
            results.append({
                "label": label,
                "confidence": f"{score:.4f}",
                "type": "mask",
                "mask": cvat,
                "points": poly.ravel().tolist(),
                "attributes": []
            })
        print(f"Filtered out {count} segments without contours")
        return results, valid_indices

    def infer(self, image, label: str, annotations_proc: list[dict]):
        """
        Infer the model on the given image and return the results.

        Parameters
        ----------
        image: PIL.Image
            Input image to process
        label: str
            Label to assign to the detected objects
        annotations_proc: list[dict]
            List of preprocessed annotations from CVAT annotations
            Each annotation is a dict with keys:
                - "id": unique identifier for the embedding
                - "type": type of the embedding (e.g., "mask", "polygon")
                - "points": points of the embedding (rle mask)
        """
        # Get embeddings from CVAT annotations
        print("Cantidad de annotations:", len(annotations_proc))


        print(1/0)
        # Preprocess the image
        img = np.array(image)
        self.mask_embeddings.clear()
        segments = self.mask_generator.generate(img)
        print("Cantidad de segmentos obtenidos:", len(segments))

        # Convert to CVAT masks
        results, valid_indices = self._segments_to_cvat_masks(segments, label)
        print("Cantidad de segmentos procesados:", len(results))

        # Retrieve embeddings
        global_emb = self.mask_generator.predictor.saved_image_embeddings
        if global_emb is not None:
            global_emb = global_emb.detach().cpu().numpy()
        mask_embs = np.array(self.mask_embeddings)[valid_indices]

        print("shape:")
        print("Embedding global:", None if global_emb is None else global_emb.shape)
        print("Embeddings de máscaras:", mask_embs.shape)

        return results#, global_emb, mask_embs


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