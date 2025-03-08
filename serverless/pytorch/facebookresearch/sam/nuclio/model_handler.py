"""
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""

import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

import matplotlib.pyplot as plt
import os

class ModelHandler:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_checkpoint = "/opt/nuclio/sam/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam_model.to(device=self.device)
        self.predictor = SamPredictor(self.sam_model)

    def _get_mean_and_std_of_each_channel(self, image):
        """
        Given an image, compute the mean and standard deviation of each channel.
        """
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        return mean, std
    
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
        new_w = orig_w * 1
        new_h = orig_h * 1
        new_x_min = int(max(cx - new_w / 2, 0))
        new_y_min = int(max(cy - new_h / 2, 0))
        new_x_max = int(min(cx + new_w / 2, w))
        new_y_max = int(min(cy + new_h / 2, h))
        return [new_x_min, new_y_min, new_x_max, new_y_max]

    def _return_to_original_image_coordinates(self, features, bbox, image_shape):
        # To map the cropped features back to the full image coordinates, we need to
        # place the cropped features into a full-size feature map.
        #
        # The full image, when processed by SAM, produces features of shape [1, 256, 64, 64].
        # Thus we create an empty tensor of that shape.
        full_feat_shape = features.shape  # [1, 256, 64, 64]
        fixed_features = torch.zeros_like(features)

        # Ensure the target region size is within the full feature map bounds
        target_h = max(1, y_max_feat - y_min_feat + 1)
        target_w = max(1, x_max_feat - x_min_feat + 1)

        # Resize the cropped features to the target size
        resized_features = torch.nn.functional.interpolate(
            features, size=(target_h, target_w), mode='bilinear', align_corners=False
        )

        # Handle potential out-of-bounds indices
        x_min_feat = max(0, x_min_feat)
        y_min_feat = max(0, y_min_feat)
        x_max_feat = min(x_max_feat, full_feat_shape[-1] - 1)
        y_max_feat = min(y_max_feat, full_feat_shape[-2] - 1)

        # Insert the resized cropped features into the full feature map
        fixed_features[:, :, y_min_feat:y_max_feat+1, x_min_feat:x_max_feat+1] = resized_features

        return fixed_features

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
        print("EJECUTANDO model_handler.py DE SAM")
        print("positive_points =", positive_points)
        print("negative_points =", negative_points)
        image_np = np.array(image)
        print("Tamaño de la imagen:", image_np.shape)
        
        # Get the mean and standard deviation of the pixel values in the SAM model.
        mean_img, std_img = self._get_mean_and_std_of_each_channel(image)
        
        print("mean_img =", mean_img, "type(mean_img) =", type(mean_img))
        print("std_img =", std_img, "type(std_img) =", type(std_img))
        self.sam_model.pixel_mean = torch.tensor(mean_img, dtype=torch.float32).view(3, 1, 1).to(self.device)
        self.sam_model.pixel_std = torch.tensor(std_img, dtype=torch.float32).view(3, 1, 1).to(self.device)
        
        self.predictor = SamPredictor(self.sam_model)
        
        # # Combine positive and negative points. If are 2 or more negative points
        # if len(negative_points) >= 2:    
        #     all_points = positive_points + negative_points
        #     bbox = self._compute_enlarged_bbox(all_points, image_np.shape)
        #     x_min, y_min, x_max, y_max = bbox
        #     print("Cropping image to bbox:", bbox)
        #     cropped_image = image_np[y_min:y_max, x_min:x_max, :]
        #     plt.imshow(cropped_image)
        #     plt.savefig("/tmp/debug_image.png")
        #     plt.close()
        # else:
        #     # If there are less than 2 negative points, use the full image.
        #     bbox = [0, 0, image_np.shape[1], image_np.shape[0]]
        #     cropped_image = image_np
        
        # ----- INIT: DEBUG -----
        cropped_image = image_np
        bbox = [0, 0, image_np.shape[1], image_np.shape[0]]
        # ----- END:  DEBUG -----
        
        # Pass the cropped image to SAM predictor.
        print("Tamaño de la imagen recortada:", cropped_image.shape)
        self.predictor.set_image(cropped_image)
        features = self.predictor.get_image_embedding()
        # print("features.shape =", features.shape)
        # for feature in features:
        #     print("feature:", feature)
            
        # # --- Return to the original image coordinates ---
        # fixed_features = self._return_to_original_image_coordinates(
        #     features, bbox, image_np.shape)
        
        # ----- INIT: DEBUG -----
        fixed_features = features
        # ----- END:  DEBUG -----
        
        print("Returning the fixed features.")
        return fixed_features

