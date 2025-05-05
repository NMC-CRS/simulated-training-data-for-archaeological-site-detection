# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:18:23 2024

@author: Claudine Gravel-Miguel

@Description: This script separates the provided tiles dataset into training/validation/testing datasets. 
If the separation is random, it uses the sklearn train_test_split function to set 80% into training, and 10% in validation and testing, respectively.
If the separtion is geographic, it uses the train_bound to separate the datasets.

"""

from sklearn.model_selection import train_test_split
import re

def extract_numbers_from_tile_format(s):
    '''
    Extracts the two coordinates from a tile name only if its name has the correct format

    Parameters
    ----------
    s : str
        This should be the name of the tile

    Returns
    -------
    bool
        Returns a list of the two numbers in the tile name if it has the proper format, and None if it doesn't.

    '''

    pattern = r'^Tile(\d+)_(\d+)\.tif$'
    match = re.match(pattern, s)
    if match:
        return match.groups()
    else:
        return None

def separate_dataset(filenames, separation_random, train_bounds):
    '''
    Separates the list from filenames into training and validation/testing datasets.
    If separation_random is set to True, the training dataset includes ~80% of the dataset, whereas the validation and testing datasets each have ~10% of the original datasets
    If separation_random is set to False, the training dataset includes tiles within the train_bounds, and the rest is divided into validation and testing datasets based on the mean X coordinates of the remaining tiles.

    Parameters
    ----------
    filenames : list
        List of tile names.
    separation_random : bool
        If the tiles are separated randomly (~80-10-10) or not.
    train_bounds : list
        xmin, ymin, xmax, and ymax around the tiles that will be used for training.

    Returns
    -------
    inputs_train : list
        List of image tile names that are used for training.
    inputs_val : list
        List of image tile names that are used for validation.
    inputs_test : list
        List of image tile names that are used for testing.

    '''
    
    if separation_random:

        # Set the random seed to make sure that we separate the paired input-targets similarly.
        random_seed = 42
        
        # Split dataset into training and validation
        train_size = 0.80  # 80:20 split at first
        
        # Randomly put the list of file names into their respective lists *80% in training and 20% in validation).
        inputs_train, inputs_val = train_test_split(
            filenames,
            random_state = random_seed,
            train_size = train_size,
            shuffle = True)
        
        # Further separate the validation dataset in two (50% validation and 50% testing)
        inputs_val, inputs_test = train_test_split(
            inputs_val,
            random_state = random_seed,
            train_size = 0.5,
            shuffle = True)

    else:
        # Create empty lists that will take the appropriate filenames
        inputs_train = []
        inputs_val_temp = []
        
        # Iterate through all filenames (tiles)
        for file in filenames:
            
            coord = extract_numbers_from_tile_format(file)
            
            if coord is None:
                
                print(f"\n'{file}' (in your first vis input tiles' folder) has an incorrect name (corrupt copy?). It has been ignored, but you should still delete it.")
                pass
            
            else:
                # Get the coordinates of each file (lower left anchor) from their name
                y_orig = float(coord[1])
                x_orig = float(coord[0])
                
                # Add the filename to its appropriate list based on its coordinates, compared to the train_bounds provided
                # Every tile within the train_bounds goes into the training dataset, everything else goes in both validation and testing datasets
                if x_orig >= train_bounds[0] and y_orig >= train_bounds[1] and x_orig < train_bounds[2] and y_orig < train_bounds[3]:
                    inputs_train.append(file)
                else:
                    inputs_val_temp.append(file)
        
        # Calculate the mean X-coordinate
        coords = [extract_numbers_from_tile_format(filename) for filename in inputs_val_temp]
        x_coords = [int(coord[0]) for coord in coords]
        mean_x = sum(x_coords) / len(x_coords)
        
        # Split the filenames into two lists based on the mean X-coordinate
        inputs_val = [filename for filename in inputs_val_temp if int(extract_numbers_from_tile_format(filename)[0]) < mean_x]
        inputs_test = [filename for filename in inputs_val_temp if int(extract_numbers_from_tile_format(filename)[0]) >= mean_x]
    
    # Return the three list of tile names
    return inputs_train, inputs_val, inputs_test

# THE END