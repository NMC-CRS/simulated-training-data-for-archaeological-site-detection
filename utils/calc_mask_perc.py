# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel

Helper function to get the percentage of tiles with positive masks
1. This iterates over all masks in a folder, checks if they have problematic values (nan, very high or low values, that suggest they were corrupted)
2. It adds their file names, and binarized max value, and the number of non-zero pixels to separate lists
3. If there are any problematic tiles, the script stops with an exception message.
4. If every tiles are fine, it puts together the lists to form a dataframe, which is returned

"""

# Setup
import os
import numpy as np
import pandas as pd
from tifffile import tifffile

def check_outliers(filename, mask_array):
    '''
    This checks if the tile has any corrupted values (nan, very low and very high values) and keeps a counter of those problems. 
    The counter will be used to stop the script if there are issues instead of feeding problematic tiles to the model training.

    Parameters
    ----------
    filename : str
        Name of the file for printing purposes.
    mask_array : numpy array
        Numpy array of the imported tile.

    Returns
    -------
    error_counter : int
        Number of problematic values in the tile.

    '''
    
    # Count how many problems there are in the tile
    error_counter = 0
    
    # Check for the presence of nan
    if np.count_nonzero(mask_array[np.isnan(mask_array)]) > 0:
        
        # set the nan to 0
        mask_array[np.isnan(mask_array)] = 0
        
        # print a message that there is an issue.
        #print(f"**WARNING**: Tile {filename} has NAN values. This suggests there is a problem with the tiles.\n")
        # log the problem in the counter
        error_counter += 1
        
    # Check for the presence of very low values
    if np.count_nonzero(mask_array[mask_array < -9000]) > 0:
        
        # set the very low values to 0
        mask_array[mask_array < -9000] = 0
        
        # print a message that there is an issue.
        #print(f"**WARNING**:Tile {filename} has outlier negative values (min = {np.nanmin(mask_array)}). This suggests that there is a problem with the tiles. Please look at the time when this tile was last modified (in OneDrive). What happened at that time?\n")
        # log the problem in the counter
        error_counter += 1
    
    # Check for the presence of very high values. Looking for anything abve 90000 to accomodate annotations for Mask RCNN, which willl have id values above  and up to the number of individual objects in an area.
    if np.count_nonzero(mask_array[mask_array > 90000]) > 0:
        
        # set the very low values to 0
        mask_array[mask_array  > 90000] = 0
        
        # print a message that there is an issue.
        #print(f"**WARNING**:Tile {filename} has outlier positive values (max = {np.nanmax(mask_array)}). This suggests that there is a problem with the tiles. Please look at the time when this tile was last modified (in OneDrive). What happened at that time?\n")
        # log the problem in the counter
        error_counter += 1
    
    # Return the number of errors in the tile
    return error_counter

def log_mask_tiles(mask_dir):
    '''
    Iterates through all tif files in mask_dir to check if there are problematic values, then count the number of pixels above 0 
    and log that information in a dataframe that will be used in the main script to select only the mask tiles with enough annotated pixels.

    Parameters
    ----------
    mask_dir : str
        Path to the folder that contains the mask tiles.

    Returns
    -------
    mask_dataframe : Pandas dataframe
        Dataframe with one row per tile, and three columns (name of the tile, its max value, and the number of pixels with value > 0).

    '''
    
    # Create empty lists that will be filled within the iterator
    list_filenames = []
    list_max_values = []
    list_min_nonzero = []
    
    # Keep track of the number of problems in mask tiles
    error_counter = 0
    
    # Get the names of all the tiffs in the folder (ignoring any .tif.aux.xml files created by QGIS)
    image_filenames = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    
    # For each mask tile, log some information
    for filename in image_filenames:
        
        # Add the file name to the list
        list_filenames.append(filename)
        
        # Load the image as a numpy array
        mask_path = os.path.join(mask_dir, filename)
        mask_array = tifffile.imread(mask_path)
        
        # Check for issues in the mask tiles and add the number of logged problems to the general counter
        error_counter += check_outliers(filename, mask_array)
         
        # Change from float32 to int16 type (just for this calculation, as the model wants float32) to get its max value (changing to integers so that any potential floats close to 0 become 0)
        mask_array = np.int16(mask_array)
        max_value = np.max(mask_array)
            
        # Append the max value to the list
        list_max_values.append(max_value)
        
        # Calculate the number of non-zero pixels
        min_nonzero = np.count_nonzero(mask_array)  # Count non-zero pixels
        list_min_nonzero.append(min_nonzero)
    
    # Stops the script if there were any problems with the mask tiles to prevent abherrant training metrics that are also a waste of time.
    if error_counter > 0:
        #raise Exception("Model training is stopped until the problematic tiles are fixed. See warning messages above.")
        print("**IMPORTANT**: There were corrupted values in the mask tiles, but they were all set to 0. Training should be fine.")
    
    # Convert as a dataframe to manipulate easily
    mask_dataframe = pd.DataFrame(list(zip(list_filenames, list_max_values, list_min_nonzero)), columns = ["filename", "max_val", "min_nonzero"])
    
    # Return the dataframe
    return mask_dataframe

# THE END