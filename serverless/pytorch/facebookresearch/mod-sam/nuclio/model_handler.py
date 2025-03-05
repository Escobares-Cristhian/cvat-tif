"""
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam_model.to(device=self.device)
        self.predictor = SamPredictor(sam_model)

    def _compute_enlarged_bbox(self, points, image_shape):
        """
        Given a list of points [[x,y], ...] and image_shape (height, width, channels),
        compute the minimal bounding box around these points, then enlarge it by a factor of 2.
        Returns bbox as [x_min, y_min, x_max, y_max], clipped to image boundaries.
        """
        h, w = image_shape[:2]
        if not points:
            # If no points are provided, return full image bbox.
            return [0, 0, w, h]
        pts = np.array(points)
        x_min = np.min(pts[:, 0])
        y_min = np.min(pts[:, 1])
        x_max = np.max(pts[:, 0])
        y_max = np.max(pts[:, 1])
        # Center of original bbox.
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        orig_w = x_max - x_min
        orig_h = y_max - y_min
        # Enlarge width and height by factor of 2.
        new_w = orig_w * 2
        new_h = orig_h * 2
        new_x_min = int(max(cx - new_w / 2, 0))
        new_y_min = int(max(cy - new_h / 2, 0))
        new_x_max = int(min(cx + new_w / 2, w))
        new_y_max = int(min(cy + new_h / 2, h))
        return [new_x_min, new_y_min, new_x_max, new_y_max]

    def handle(self, image, positive_points, negative_points):
        """
        Processes the input PIL image by:
         1. Converting to a NumPy array.
         2. Combining positive and negative points.
         3. Computing the 2x-enlarged minimal bounding box.
         4. Cropping the image to that box.
         5. Passing the cropped image to SAM predictor.
        Returns the image embedding and the crop bounding box.
        """
        image_np = np.array(image)
        # Combine positive and negative points.
        all_points = positive_points + negative_points
        bbox = self._compute_enlarged_bbox(all_points, image_np.shape)
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image_np[y_min:y_max, x_min:x_max]
        self.predictor.set_image(cropped_image)
        features = self.predictor.get_image_embedding()
        return features, bbox

