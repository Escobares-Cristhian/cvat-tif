"""
# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT
"""

import numpy as np
import torch
import torch.nn.functional as F
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
            return [0, 0, w, h]
        pts = np.array(points)
        x_min = np.min(pts[:, 0])
        y_min = np.min(pts[:, 1])
        x_max = np.max(pts[:, 0])
        y_max = np.max(pts[:, 1])
        
        # Compute the center of the original bbox.
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

    def _return_to_original_image_coordinates(self, features, crop_bbox, original_image_shape):
        """
        Transform the feature map from the cropped image back to the original image space.
        
        Parameters:
            features (torch.Tensor): The feature map output from SAM's image_encoder with shape [1, 256, 64, 64].
            crop_bbox (list): The bounding box used to crop the image [x_min, y_min, x_max, y_max].
            original_image_shape (tuple): The shape of the original image (height, width).

        Returns:
            torch.Tensor: The feature map resized and repositioned in the original image space.
        """
        
        # Extract crop bounding box and original dimensions
        x_min, y_min, x_max, y_max = crop_bbox
        orig_h, orig_w = original_image_shape[:2]
        
        # Compute the corresponding feature map coordinates for the crop region.
        # The feature map has a fixed resolution (64x64)
        feature_x_min = int(round((x_min / orig_w) * 64))
        feature_y_min = int(round((y_min / orig_h) * 64))
        feature_x_max = int(round((x_max / orig_w) * 64))
        feature_y_max = int(round((y_max / orig_h) * 64))
        
        # Compute the target size for the cropped region in feature space
        target_fH = feature_y_max - feature_y_min
        target_fW = feature_x_max - feature_x_min
        
        # Resize the feature map from the cropped image to the target size
        resized_features = F.interpolate(features, size=(target_fH, target_fW), mode='bilinear', align_corners=False)
        
        # Create a full-sized feature map for the original image (fixed size 64x64)
        full_features = torch.zeros((1, 256, 64, 64), device=features.device)
        
        # Place the resized features in the correct position within the full feature map
        full_features[:, :, feature_y_min:feature_y_max, feature_x_min:feature_x_max] = resized_features
        
        return full_features

    
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
        print(f"positive_points = {positive_points}")
        print(f"negative_points = {negative_points}")
        image_np = np.array(image)
        print(f"Tamaño de la imagen: {image_np.shape}")
             
        # Combine positive and negative points. If are 2 or more points
        all_points = positive_points + negative_points
        if len(all_points) >= 2:
            bbox = self._compute_enlarged_bbox(all_points, image_np.shape)
            x_min, y_min, x_max, y_max = bbox
            print(f"Cropping image to bbox: {bbox}")
            cropped_image = image_np[y_min:y_max, x_min:x_max, :]
            plt.imshow(cropped_image)
            plt.savefig("/tmp/debug_image.png")
            plt.close()
        else:
            # If there are less than 2 negative points, use the full image.
            bbox = [0, 0, image_np.shape[1], image_np.shape[0]]
            cropped_image = image_np
        
        # Get the mean and standard deviation of the pixel values in the SAM model.
        mean_img, std_img = self._get_mean_and_std_of_each_channel(cropped_image)
        
        print(f"mean_img = {mean_img}, type(mean_img) = {type(mean_img)}")
        print(f"std_img = {std_img}, type(std_img) = {type(std_img)}")
        self.sam_model.pixel_mean = torch.tensor(mean_img, dtype=torch.float32).view(3, 1, 1).to(self.device)
        self.sam_model.pixel_std = torch.tensor(std_img, dtype=torch.float32).view(3, 1, 1).to(self.device)
        
        self.predictor = SamPredictor(self.sam_model)
        
        # Pass the cropped image to SAM predictor.
        print(f"Tamaño de la imagen recortada: {cropped_image.shape}")
        self.predictor.set_image(cropped_image)
        features = self.predictor.get_image_embedding()
        print(f"features.shape = {features.shape}")
        
        return features

