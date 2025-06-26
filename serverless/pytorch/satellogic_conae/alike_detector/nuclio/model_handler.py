import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor
from skimage.measure import find_contours, approximate_polygon
import torch
import types
# from pycocotools import mask as maskUtils
import time
import math

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

def rle_to_mask2d(data, start_val=0):
    """
    Decode run‐length encoding (RLE) + box tail back into the 2D mask crop.

    Parameters
    ----------
    data : Sequence[int]
        A sequence whose last four entries are [xtl, ytl, xbr, ybr] (inclusive
        pixel coords), and whose preceding entries are RLE counts:
        [#zeros, #ones, #zeros, #ones, …].
    start_val : {0,1}, default 0
        Which value the first run‐length corresponds to (usually 0).

    Returns
    -------
    mask_2d : ndarray of shape (h, w), dtype uint8
        The reconstructed binary mask crop.

    Raises
    ------
    ValueError
        If the total run‐length doesn’t match the box area.
    """
    # 1) unpack box coords and compute size
    xtl, ytl, xbr, ybr = map(int, data[-4:])
    h = ybr - ytl + 1
    w = xbr - xtl + 1

    # 2) pull off the RLE counts
    rle = list(map(int, data[:-4]))
    total = sum(rle)
    if total != h * w:
        raise ValueError(f"Expected total run-length {h*w}, got {total}")

    # 3) decode the runs
    flat = []
    val = start_val
    for length in rle:
        flat.extend([val] * length)
        val = 1 - val

    # 4) reshape to 2D crop
    mask_2d = np.array(flat, dtype=np.uint8).reshape((h, w))
    return xtl, ytl, h, w, mask_2d


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

        # Initialize encoder:
        self.encoder = SAM2ImagePredictor(
            build_sam2(self.model_cfg, self.sam_checkpoint, device=self.device, apply_postprocessing=False),
            device=self.device
        )


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

    def image_to_embedding(self, image):
        self.encoder.set_image(image)
        embedding = self.encoder.get_image_embedding()
        return embedding

    def annotations_to_embeddings(self, full_image, annotations):
        embeddings = []
        for annotation in annotations:
            if str(annotation["type"]) != "mask":
                raise ValueError(f"Unsupported annotation type: {annotation['type']}. Only 'mask' is supported.")
            # Convert CVAT mask to 2D mask and box
            x0, y0, w, h, mask_2d = rle_to_mask2d(annotation["points"])
            # Extract the region of interest from the image
            cut_image = full_image[y0:y0+h, x0:x0+w]                # Crop the image to the bounding box
            # cut_image = np.where(mask_2d[..., None], cut_image, 0)  # Apply the mask to the image
            cut_image = np.where(np.dstack([mask_2d.T]*3), cut_image, 0)  # Apply the mask to the image ->  (224,121,3) (121,224,3) error
            # DEBUG: En vez de "0", capaz conviene usar un número aleatorio para que SAM no detecte el fondo como un objeto

            t1 = time.time()
            embedding = self.image_to_embedding(cut_image)
            t2 = time.time()
            print(f"Embedding time: {t2 - t1:.4f} seconds")

            print(f"embedding type: {type(embedding)}") # -> <class 'torch.Tensor'>
            print(f"embedding shape: {embedding.shape if getattr(embedding, 'shape', None) else 'No shape attribute'}")
            embeddings.append(embedding.detach().cpu().numpy())
        return embeddings

    def cosine_similarity(self, vec1, vec2):
        # 1. Dot product
        dot = sum(a * b for a, b in zip(vec1, vec2))
        # 2. Norms
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        # 3. Handle zero-vector edge case
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 4. Cosine similarity
        return dot / (norm1 * norm2)

    def get_emb_similatiry(self, emb1, emb2):
        """
        Calculate the cosine similarity between two embeddings.

        Parameters
        ----------
        emb1: torch.Tensor
            First embedding tensor
        emb2: torch.Tensor
            Second embedding tensor

        Returns
        -------
        float
            Cosine similarity between the two embeddings
        """
        # emb1, emb2: e.g. (1, C, H, W) or (C, H, W)
        # squeeze out batch‐dim if present
        if emb1.ndim == 4 and emb1.shape[0] == 1:
            e1 = emb1[0]
            e2 = emb2[0]
        else:
            e1 = emb1
            e2 = emb2

        # now e1, e2 have shape (C, H, W)
        # dot‐product over channel dim → map of shape (H, W)
        num = np.sum(e1 * e2, axis=0)
        # norms over channel dim → two maps (H, W)
        norm1 = np.linalg.norm(e1, axis=0)
        norm2 = np.linalg.norm(e2, axis=0)
        denom = norm1 * norm2
        # safe divide
        cos_map = np.zeros_like(num)
        valid = denom > 0
        cos_map[valid] = num[valid] / denom[valid]

        # finally, collapse to a single score
        return float(np.median(cos_map))

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
        # Preprocess the image
        img = np.array(image)

        # Get embeddings from CVAT annotations
        print("Cantidad de annotations:", len(annotations_proc))

        if len(annotations_proc) > 0:
            print("Obteniendo embeddings de las anotaciones...")
            self.annot_embeddings = self.annotations_to_embeddings(img, annotations_proc)
            print("Cantidad de embeddings obtenidos:", len(self.annot_embeddings))
        else:
            print("No se han proporcionado anotaciones para obtener embeddings.")
            self.annot_embeddings = []

        # Process the image with the mask generator
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

        print(f"shape mask_embs: {mask_embs.shape}")

        print("shape:")
        print("Embedding global:", None if global_emb is None else global_emb.shape)
        print("Embeddings de máscaras:", mask_embs.shape)

        # Get mean annot_embeddings:
        if len(self.annot_embeddings) > 0:
            mean_annot_emb = np.mean(np.array(self.annot_embeddings), axis=0)
            print(f"shape mean_annot_emb: {mean_annot_emb.shape}")

        # # Get global embedding of mask (WARNING):
        # if len(mask_embs) > 0:
        # # DEBUG: uso el primer mask-token embedding de cada máscara. Y paso de (1, 256) -> (1, 256, 1, 1)
        #     mask_embs = np.array([[emb[0].reshape(256,1,1)] for emb in mask_embs])
        #     mask_embs = np.array([global_emb * emb for emb in mask_embs])
        #     print(f"shape mask_embs after global emb: {mask_embs.shape}")

        # Get real embedding of masks:
        if len(mask_embs) > 0:
            t1 = time.time()
            mask_embs = [res["mask"] for res in results] # Extract binary masks and extent from the masks
            extent_embs = [res[-4:] for res in mask_embs] # Extract extent from the masks
            mask_embs = [np.array(emb[:-4]) for emb in mask_embs] # Remove extent from the masks

            # # Convert CVAT mask to 2D mask and box
            # x0, y0, w, h, mask_2d = rle_to_mask2d(annotation["points"])
            # # Extract the region of interest from the image
            # cut_image = full_image[y0:y0+h, x0:x0+w]                # Crop the image to the bounding box
            # # cut_image = np.where(mask_2d[..., None], cut_image, 0)  # Apply the mask to the image
            # cut_image = np.where(np.dstack([mask_2d.T]*3), cut_image, 0)  # Apply the mask to the image ->  (224,121,3) (121,224,3) error
            # # DEBUG: En vez de "0", capaz conviene usar un número aleatorio para que SAM no detecte el fondo como un objeto

            print(f"shape img: {img.shape}")
            print(f"example extent_embs[0]: {extent_embs[0]}")
            print(f"shape img cut test 0: {img[extent_embs[0][0]:extent_embs[0][2]+1, extent_embs[0][1]:extent_embs[0][3]+1, :].shape}")
            print(f"shape embs reshape test 0: {mask_embs[0].reshape(extent_embs[0][2]-extent_embs[0][0]+1, extent_embs[0][3]-extent_embs[0][1]+1).shape}")
            mask_embs = [
                    img[e[1]:e[3]+1, e[0]:e[2]+1, :]
                    * emb.reshape(e[3]-e[1]+1, e[2]-e[0]+1)[..., None]
                    for emb, e in zip(mask_embs, extent_embs)]

            t2 = time.time()
            print(f"Mask embedding time: {t2 - t1:.4f} seconds")

            import matplotlib.pyplot as plt
            plt.imshow(mask_embs[0])
            plt.savefig("mask_embs_0.png")
            plt.close()

            t1 = time.time()
            mask_embs = [self.image_to_embedding(img_tmp).detach().cpu().numpy() for img_tmp in mask_embs]
            t2 = time.time()
            print(f"Mask embedding time (image_to_embedding): {t2 - t1:.4f} seconds")


        # Compare with annot_embeddings:
        if len(mask_embs) > 0:
            print(f"shape mask_embs[0]: {mask_embs[0].shape}")
            sim_embs = np.array([
                self.get_emb_similatiry(emb, mean_annot_emb) for emb in mask_embs
            ])

            # Imprimo histograma con las similitudes
            import matplotlib.pyplot as plt
            plt.hist(sim_embs, bins=50, alpha=0.7, color='blue')
            plt.title('Histogram of Similarities')
            plt.xlabel('Similarity')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.savefig("similarities_histogram.png")
            plt.close()

            # # Select the highest similarity
            # highest_sim_index = np.argmax(sim_embs)
            # highest_sim_value = sim_embs[highest_sim_index]
            # print(f"Highest similarity index: {highest_sim_index}, value: {highest_sim_value:.4f}")
            # results = [results[highest_sim_index]]
            # # Select 10 highest similarities
            # highest_sim_indices = np.argsort(sim_embs)[-10:][::-1]
            # print(f"Highest similarity indices: {highest_sim_indices}, values: {[sim_embs[i] for i in highest_sim_indices]}")
            # Select similarity over 0.85
            highest_sim_indices = np.where(sim_embs > 0.85)[0]
            print(f"Similarities over 0.85 found: {len(highest_sim_indices)} out of {len(sim_embs)}")
            print(f"Highest similarity indices: {highest_sim_indices}, values: {[sim_embs[i] for i in highest_sim_indices]}")
            if len(highest_sim_indices) == 0:
                print("No similarities over 0.85 found, returning empty results.")
                return []

            results = [results[i] for i in highest_sim_indices]


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