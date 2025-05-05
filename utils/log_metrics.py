#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:51:18 2024

@author: cgravelmiguel

@ Description: Script to Create a csv file to log metrics (or write to an existing one), identify the next empty row, and enter new info to it

"""

import os
import pandas as pd
from datetime import datetime
import platform
import re

def populate_params(output_path, structure, backbone, lr_variable, loss_fun, batch_size, vis1, vis2, vis3, threshold, train_n, val_n, test_n, remove_overlap, buffer_size, mask_folder_name, train_bounds, n_epochs, time_stamp):
    '''
    This looks for a Metrics_log.csv in the CNN_output folder. 
    If there is one, it looks for the next empty row and populate it with the given parameter values
    If there is none, it creates a new CSV and populate it with the given parameter values

    Parameters
    ----------
    output_path : str
        Path to the CNN_output folder in which the CSV is or will be saved.
    structure : str
        Name of the model structure (e.g., UNet, MaskRCNN, or FasterRCNN).
    backbone : str
        Name of the backbone used for training.
    lr_variable : bool
        If the learning rate is updating when val_loss stagnates (True) or kept constant at 0.001 (False).
    loss_fun : str
        Name of the loss function to optimize (e.g., dice, iou, focal).
    batch_size : int
        Number of images that will be uploaded to the model at the same time.
    vis1 : str
        Name of the first visualization.
    vis2 : str
        Name of the second visualization.
    vis3 : str
        Name of the third visualization.
    threshold : int
        For pre-processing. This removes any mask images that have less than the threshold's number of positive's values.
    train_n : int
        Number of tiles used for training.
    val_n : int
        Number of tiles used for validation.
    test_n : int
        Number of tiles used for testing.
    remove_overlap : bool
        If we clean the datasets to remove any overlap between them.
    buffer_size : int
        Buffer size (in meters) around the annotated object. This is to find the correct folder to upload.
    mask_folder_name : str
        Name of the folder that holds the masks folders. This is to find the correct folder to upload.
    train_bounds : list
        xmin, ymin, xmax, and ymax around the tiles that will be used for training.
    n_epochs : int
        Number of epochs to train the model. The model sees all the training images in each epoch.
    time_stamp : int
        The timestamp used to distinguish between trainings (end of filename).

    Returns
    -------
    None.

    '''
    # Define the header and parameter values
    columns = [
        'Date', 'Platform', 'Computer', 'Structure', 'Frozen_seed', 'Backbone',
        'Pretrained_weights', 'Learning_rate', 'Optimizer', 'Loss', 'Batch_size',
        'Vis1', 'Vis2', 'Vis3', 'Train_threshold', 'Train_ds_n',
        'Val_ds_n', 'Test_ds_n', 'No_overlap', 'Mask_buffer', 'Mask_name', 'Post_threshold',
        'Train_%', 'Train_bounds', 'N_epochs', 'Duration_min', 'Train_loss',
        'Train_recall', 'Train_precision', 'Train_F1', 'Train_MCC', 'Train_acc',
        'Val_loss', 'Val_recall', 'Val_precision', 'Val_F1', 'Val_MCC', 'Val_acc',
        'Test_recall', 'Test_precision', 'Test_F1', 'Test_MCC', 'Object_recall',
        'Object_precision', 'Object_F1', 'Object_TP', 'Object_FN', 'Object_FP',
        'Tag', 'Notes'
    ]    

    row_values = {
        'Date': datetime.now().strftime('%Y-%m-%d'), # today's date
        'Platform': "Pytorch",
        'Computer': platform.node(), # identifies the computer name
        'Structure': structure,
        'Frozen_seed': "Yes",
        'Backbone': backbone,
        'Pretrained_weights': "Yes", 
        'Learning_rate': lr_variable, 
        'Optimizer': "AdamW", 
        'Loss': loss_fun, 
        'Batch_size': batch_size,
        'Vis1': vis1, 
        'Vis2': vis2, 
        'Vis3': vis3, 
        'Train_threshold': threshold, 
        'Train_ds_n': train_n,
        'Val_ds_n': val_n, 
        'Test_ds_n': test_n, 
        'No_overlap': remove_overlap,
        'Mask_buffer': buffer_size, 
        'Mask_name': mask_folder_name, 
        'Train_%': round(train_n / (train_n + val_n + test_n), 2), 
        'Train_bounds': train_bounds, 
        'N_epochs': n_epochs, 
        'Tag': time_stamp
    }    
    
    # Define the file name and path
    file_name = 'Metrics_log.csv'
    file_path = os.path.join(output_path, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_name}' not found. Creating a new file.")
        # Create a new DataFrame with the specified columns
        df = pd.DataFrame(columns=columns)
        # Add the first row with the provided values
        new_row = {col: row_values.get(col, None) for col in columns}  # Fill missing columns with None
        df = pd.DataFrame([new_row], columns=columns)
    else:
        print(f"File '{file_name}' exists. Reading the file.")
        # Read the existing CSV
        df = pd.read_csv(file_path)
        # Check for the first empty row
        mask = df['Date'].isna() | (df['Date'] == '')  # Assuming 'Date' is required for every row
        if mask.any():
            first_empty_row = mask.idxmax()
            print(f"Found empty row at index {first_empty_row}. Populating values.")
            for col, value in row_values.items():
                df.loc[first_empty_row, col] = value
        else:
            print("No empty rows found. Adding a new row.")
            # Add a new row at the end
            new_row = {col: row_values.get(col, None) for col in columns}  # Fill missing columns with None
            df = pd.concat([df, pd.DataFrame([new_row], columns=columns)], ignore_index=True)

    # Save the DataFrame back to CSV
    df.to_csv(file_path, index=False)
    print(f"Updated DataFrame saved to '{file_path}'.")

def populate_metrics(output_path, filename, target_column, new_value):
    '''
    Populates a specific cell from the Metrics log CSV.

    Parameters
    ----------
    output_path : str
        Path to the CNN_output folder in which the CSV is or will be saved.
    filename : str
        Name that is used for weights file, the folder that holds predicted tiles, the compiled geotiff, and the compiled shapefile.
    target_column : str
        Name of the column to populate
    new_value : varies
        New value to populate. Can be a number, a list, a boolean, a string...

    Returns
    -------
    None.

    '''

    # Define the file name and path
    file_name = 'Metrics_log.csv'
    file_path = os.path.join(output_path, file_name)    

    # Read the CSV file
    df = pd.read_csv(file_path)    

    # Get the filename tag from filename
    last_number = re.search(r'_(\d+)$', filename)
    filename_tag = last_number.group(1)

    # Identify the row number associated with that filename
    filename_tag_int = int(filename_tag)
    row_index = df[df['Tag'] == filename_tag_int].index
    
    if not row_index.empty:
        # If the value is found, update the target column for the identified row
        row_index = row_index[0]  # Get the first matching row index
        df.loc[row_index, target_column] = new_value  # Update the target column
    else:
        print(f"Value '{filename_tag}' not found in column 'Tag'.")
    
    # Save the modified DataFrame back to the CSV file
    df.to_csv(file_path, index=False)
    
# THE END