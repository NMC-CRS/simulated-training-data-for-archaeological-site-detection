# -*- coding: utf-8 -*-

'''
@author: Claudine Gravel-Miguel, from code by https://haochen23.github.io/2020/06/fine-tune-mask-rcnn-pytorch.html

@description: This formats the dataset to fit the Mask RCNN requirements

'''

# Import the required packages
import torch
from torch.utils.data import Dataset
from skimage.io import imread  # this is from the scikit-learn library
import numpy as np

class MaskBoxDataset(Dataset):
    '''
    A Pytorch Dataset class to load the tiles of the 3 bands, combine them to form 3D tiles, and compute the bounding boxes from their associated mask. 
    Each tile is formatted separately.
    
    Parameters
    ----------
    inputs_dim1 : list
        List of all tiles of the first visualization type (can be training, validation, or testing datasets)
    inputs_dim2 : list
        List of all tiles of the second visualization type (can be training, validation, or testing datasets)
    inputs_dim3 : list
        List of all tiles of the third visualization type (cna be training, validation, or testing datasets)
    targets : list
        List of the mask tiles associated with the dataset
    transform : function
        Augmentation function using albumentations library
    
    Returns
    -------
    image_transformed : Pytorch tensor
        The 3D transformed tile with the following dimensions (C, H, W)
    target : dictionary
        Dictionary that holds the coordinates of the bounding boxes, their labels, the tensor of the mask, a unique identifier, and the area of the boxes

    '''
    
    def __init__(self,
                 inputs_dim1: list,
                 inputs_dim2: list,
                 inputs_dim3: list,
                 targets: list,
                 transform = None
                 ):
        self.inputs_dim1 = inputs_dim1
        self.inputs_dim2 = inputs_dim2
        self.inputs_dim3 = inputs_dim3
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.inputs_dim1)
    
    def __getitem__(self, index):
        
        # Select the sample
        input_dim1_ID = self.inputs_dim1[index]
        input_dim2_ID = self.inputs_dim2[index]
        input_dim3_ID = self.inputs_dim3[index]
        target_ID = self.targets[index]

        # Load input and target as numpy arrays
        image_dim1 = imread(input_dim1_ID)
        image_dim2 = imread(input_dim2_ID)
        image_dim3 = imread(input_dim3_ID)
        mask = imread(target_ID)
               
        # Combine the 3 images into one (to form a 3 band tile)
        image = np.stack((image_dim1, image_dim2, image_dim3), 2)
        
        # Get the number of different objects in the image
        # Instances are encoded as different colors
        obj_ids = np.unique(mask)
        
        # First id is the background, so remove it
        obj_ids = obj_ids[1:]
        
        # Separate the 1D mask into an array that has as many dimensions as there are objects,
        # and each dimension holds boolean values showing where that object's values are.
        masks = mask == obj_ids[:, None, None]
                
        ## Transformations
        # Feed both image and mask to transform to ensure that the pair (image-mask) is transformed in the same way
        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask) 
            image_transformed = transformed['image']
            masks_transformed = transformed['mask']
        
            # Update the global masks and image variables with the transformed versions.
            masks = masks_transformed == obj_ids[:, None, None]
            image = image_transformed
        
        # Get bounding box coordinates for each mask
        num_objs = len(masks)
        bboxes = []
        for i in range(num_objs):
            # Get the x and y values of pixels with values in them (x and y are separate dimensions)                
            pos = np.where(masks[i]) 
            
            # If there is no more mask after transformation, it creates a small bounding box around one pixel in a corner
            if pos[0].size == 0 or pos[1].size == 0:
                bboxes.append([0, 0, 1, 1])
            else: 
                # Extract the min and max of x and y values, which are the 4 corners of the bounding box.
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
            
                # If the annotation is only on an edge, this expands the annotation just enough to prevent breaking the model
                if xmax <= xmin or ymax <= ymin:
                    if xmin == 0 and xmax == 0:
                        xmax = 1
                    if ymin == 0 and ymax == 0:
                        ymax = 1
                    if xmin == xmax:
                        xmin = xmax - 1
                    if ymin == ymax:
                        ymin = ymax - 1
            
                # Append the box to the list
                bboxes.append([xmin, ymin, xmax, ymax])
        
        # Transform the numpy array into a Pytorch float32 tensor with the correct dimension order (C, H, W)
        image_transformed = torch.as_tensor(image, dtype = torch.float32)
        image_transformed = image_transformed.permute(2,0,1)
        
        # Format bounding boxes, masks, and other info for the target dictionary
        boxes = torch.as_tensor(bboxes, dtype=torch.float32)

        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([index])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Return the 3-band Pytorch tensor image and its annotation target dictionary
        return image_transformed, target
    
# THE END