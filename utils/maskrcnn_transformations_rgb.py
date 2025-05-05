# -*- coding: utf-8 -*-

"""

@author: Claudine Gravel-Miguel

@description: Helper function to augment the training images and masks
This does not augment the bounding boxes, because those are calculated from the mask after augmentation (in maskrcnn_dset_rgb.py)

"""

# Import required libraries
import albumentations as A


def collate_fn(batch):
    '''
    This creates the batch (it's the glue that batches images together)

    Parameters
    ----------
    batch : list
        List of all the samples.

    Returns
    -------
    Tuple
        Returns a tuple of the zipped batch.

    '''
    
    return tuple(zip(*batch))

def train_augmentation(im_size):
    '''
    Performs transformations on uploaded images. Each transformations has a certain probability of happening.
    The transformations are done one after the other, on the array resulting from the previous transformation (if transformed)

    Parameters
    ----------
    im_size : int
        Size of the input (height or width).

    Returns
    -------
    Albumentation Compose workflow
        The workflow that actually runs the images through the transformation when they are loaded to the model.

    '''
    
    train_transform = [ 
        A.Blur(p = .25),
        A.HorizontalFlip(p = .50),
        A.VerticalFlip(p = .50),
        A.Rotate(limit = 90, p = .50),
        A.Resize(im_size, im_size), # makes sure that all tiles are of the correct format
    ]
    return A.Compose(train_transform)

# THE END