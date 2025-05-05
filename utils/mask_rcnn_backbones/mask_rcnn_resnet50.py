#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:31:41 2024

@author: Claudine Gravel-Miguel put this document together, but the code is mostly taken from the PyTorch tutorial on how to use Mask RCNN: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

@description: This imports and formats the correct Mask RCNN structure and backbone

"""

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes):
    '''
    Imports the model structure and the ResNet50 pretrained weights, sets the number of layers that will learn (remaining are frozen), 
    and replaces the head of the model with a new one that fits the new object we are training on.
    
    

    Parameters
    ----------
    num_classes : int
        Number of different classes the model will learn (should always be 2).

    Returns
    -------
    model : PyTorch model
        Mask RCNN model with loaded parameters and corrected structure.

    '''
    
    # Load an instance segmentation model pre-trained on COCO and set the number of its layers that will learn (1). The rest are frozen.
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                 trainable_backbone_layers = 1)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model
