#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:18:42 2023

@author: Claudine Gravel-Miguel

@description: This script holds most of the functions that are used to: 
1. Upload a pretrained model
2. Run new tiles (or the testing dataset) through that pretrained model, 
2. Merge the tiles created into one raster saved as a geotiff, vectorize the raster, and save that vector as a shapefile
3. Calculate the metrics per objects by comparing the raster prediction with the annotated prediction (as shp) if applicable

Its two main functions (main_with_metrics and main_without_metrics) were kept separate because they have different variable requirements.

"""

# load the packages required
import rasterio
from rasterio.merge import merge

import torch
import os
from skimage.io import imread
import numpy as np
from torchvision import transforms

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely import geometry
from shapely.affinity import affine_transform

# Import helper functions already created
import calculate_metrics
import log_metrics as logmet

# Define which device to use based on the computer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(image_dim1, image_dim2, image_dim3):
    '''
    Combine the 3 dimensions provided into a 3D array, transform into a tensor, and format its dimensions appropriately for the model

    Parameters
    ----------
    image_dim1 : numpy array
        The first visualization tile imported as numpy array.
    image_dim2 : numpy array
        The second visualization tile imported as numpy array.
    image_dim3 : numpy array
        The third visualization tile imported as numpy array.

    Returns
    -------
    im : Pytorch tensor
        Tensor of dimensions [B, C, H, W] where B stands for the batch size.

    '''
    
    # This is a torchvision transform to transform the numpy array into a Pytorch tensor
    tfms = transforms.Compose([
        transforms.ToTensor()
    ])    

    # Format the images to fit what we trained the model on.
    im = np.stack((image_dim1, image_dim2, image_dim3), 2)
    im = tfms(im)
    
    # Add the first dimension (batch size)
    im = im.unsqueeze(0) 
    
    # Return the pytorch tensor
    return im

def export_bounding_boxes(geo_dict, boxes, scores, tile_profile):
    '''
    Transform the predicted bounding boxes into a format that can be read by other GIS programs. 
    Those are then appended to the geo_dict.

    Parameters
    ----------
    geo_dict : dictionary
        Dictionary of the reformated predicted bounding boxes with their ids and scores.
    boxes : Numpy array
        Array of bounding box coordinates (each box is a sub array).
    scores : Numpy array
        Array of the scores associated with each box.
    tile_profile : crs profile
        CRS profile of the tile the predictions are from. This will be assigned to the bounding boxes to correctly place them in 3D GIS data

    Returns
    -------
    geo_dict : dictionary
        Dictionary of the reformated predicted bounding boxes with their ids and scores (updated with next boxes).

    '''

    for i in range(len(boxes)):
        formatted_coord = geometry.box(*boxes[i]) # transform the xmin, ymin, xmax, ymax coordinates into GIS coordinates
        polygon_geom = Polygon(formatted_coord)
        
        # Rasterio's affine matrix is a b c d e f, where c and f correspond to the xoff and yoff of shapely's affine matrix.
        # so I need to reorder the matrix.
        ras_m = tile_profile['transform']
        shapely_matrix = [ras_m[0], ras_m[1], ras_m[3], ras_m[4], ras_m[2], ras_m[5]]
        
        # Transform the pixel coordinates to their GIS coordinates
        polygon_geom_tr = affine_transform(polygon_geom, shapely_matrix)
        
        # Update the lists associed with each key
        geo_dict["ids"].append(i)
        geo_dict["geometry"].append(polygon_geom_tr)
        geo_dict["score"].append(round(scores[i], 3))
    
    # Return the dictionary
    return geo_dict

def postprocess_MaskRCNN(outputs):
    '''
    Filter the predicted masks and bounding boxes based on the score (keep only scores > 0.5)
    Binarize the mask and transform into a 1 dimension float32 numpy 

    Parameters
    ----------
    outputs : dictionary
        Dictionary that holds tensors of masks, bounding boxes, and scores for each predicted object.

    Returns
    -------
    boxes : numpy array
        Bounding box coordinates.
    scores : numpy array
        Scores of each prediction (0-1).
    masks : numpy array
        Array of dimension [H, W] and float32.

    '''

    # Move the outputs to CPU    
    outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
    output = outputs[0]
    
    # Keep only the predictions above a certain threshold
    boxes = output['boxes'][output['scores'] > 0.5].detach().numpy()
    masks = output['masks'][output['scores'] > 0.5].detach().numpy()
    scores = output['scores'][output['scores'] > 0.5].detach().numpy()
    
    masks[masks > 0.5] = 1
    masks[masks < 1] = 0
    
    # If there are no predictions, replace with an array of 0s
    if masks.shape[0] == 0:
        masks = np.zeros((256,256), dtype = 'float32')
    # Take the max value at each pixel (if there are more than one)
    else:
        masks = masks.max(axis = 0)
        masks = masks.squeeze()
    
    return boxes, scores, masks

def check_dict(dictionary, id1, id2, last_group_n):
    '''
    Uses the two ids to search the dictionary for previous overlaps (key:value represents the id of a polygon and its group id)
    If any of the two ids is already in the dictionary, adds any missing key to the already created group id
    If none of the ids are already in the dictionary, adds the new keys and assign to a new group id.

    Parameters
    ----------
    dictionary : dictionary
        Dictionary where keys are polygon ids and values are their group id.
    id1 : int
        The index number assigned to the first polygon in the overlapping pair.
    id2 : int
        The index number assigned to the second polygon in the overlapping pair.
    last_group_n : int
        The max group id in the dictionary (to create a new one where necessary).

    Returns
    -------
    dictionary : dictionary
        Dictionary where keys are polygon ids and values are their group id (updated if/as necessary).

    '''
    
    # First, if both keys are already there, there is nothing to add.
    if id1 in dictionary.keys() and id2 in dictionary.keys():
        #print(f'Both {id1} and {id2} are already in dict')
        pass
    # If one or the other is already in, the new key is added with the group id of the old one
    elif id1 in dictionary.keys() and id2 not in dictionary.keys():
        #print(f'{id1} is in, but {id2} is not')
        dictionary.update({id2:dictionary.get(id1)})
    elif id1 not in dictionary.keys() and id2 in dictionary.keys():
        #print(f'{id1} is not in, but {id2} is')
        dictionary.update({id1:dictionary.get(id2)})
    else:
        #print(f'{id1} and {id2} are NOT in dict')
        dictionary.update({id1:last_group_n + 1})
        dictionary.update({id2:last_group_n + 1})
    
    return dictionary

def merge_overlapping_bboxes(polygons):
    '''
    Calculate the max score of all overlapping polygons, merge those polygons into one with that max score.

    Parameters
    ----------
    polygons : geopandas dataframe
        Dataframe that holds the geometry of the predicted bounding boxes including their score.

    Returns
    -------
    intersects_diss : geopandas dataframe
        Dataframe that holds the geometry of the merged bounding boxes including the max score for each.

    '''
    
    # Move the ID to avoid zeros
    polygons["ids"] = polygons.index + 1
    
    # Join the geodatabase with itself to see which polygons overlap
    over_polys = polygons.overlay(polygons, how="intersection", keep_geom_type = True).reset_index()
    
    # Sort by coordinates to keep polygons together
    over_polys["max_score"] = over_polys[["score_1", "score_2"]].max(axis = 1)

    # Create a unique id for overlapping combinations
    combinations = pd.concat([over_polys["ids_1"], over_polys["ids_2"]], axis=1)
    combinations = combinations[combinations['ids_1'] != combinations['ids_2']]
    combinations = combinations.sort_values(["ids_1", "ids_2"], ascending=[True, False])
    
    # Create a temporary dictionary that will record group ids of overlapping polygons
    groups = {0:0}
    
    for _, row in combinations.iterrows():
        # Merge groups based on each row
        max_group_n = max(groups.values())
        groups = check_dict(groups, row['ids_1'], row['ids_2'], max_group_n)

    # Pivot to long format so that we have each possible id, set column name, and remove duplicates
    comb_long = pd.concat([combinations["ids_1"], combinations["ids_2"]])
    comb_long = comb_long.rename("ids")
    comb_long = comb_long.to_frame()
    comb_long = comb_long.drop_duplicates(ignore_index=True)

    # Then, assign the group_id to each id
    comb_long["group_id"] = comb_long["ids"].map(groups)

    # Join comb_long with the over_polys geodf to get that unique_id associated with overlaps
    over_polys_wt_id = pd.merge(over_polys, comb_long, left_on = "ids_1", right_on = "ids")

    # Dissolve the overlapping polygons using the group id
    intersects_diss = over_polys_wt_id.dissolve(by="group_id", aggfunc="max").reset_index()

    # Clean up removing variables we no longer need
    intersects_diss = intersects_diss.drop(["ids_1", "ids_2", "score_1", "score_2", "index"], axis = 1)
    intersects_diss

    # Get the non-overlapping
    non_overlapping = pd.merge(polygons, comb_long, on=["ids"], how="outer", indicator=True).query('_merge =="left_only"')
    non_overlapping = non_overlapping.drop(["_merge"], axis = 1)
    non_overlapping = non_overlapping.rename(columns = {"score": "max_score"})

    # Merge the two together back again
    intersects_diss = pd.concat([intersects_diss, non_overlapping])
    
    return intersects_diss

def predict_on_new_tiles(model, filename, tilenames, inputs_test_dim1, inputs_test_dim2, inputs_test_dim3, output_path):
    '''
    Iterate through all the files in the filenames list. For each filename, import the three visualizations as numpy arrays.
    Call functions to format the 3 bands of visualization into a 3D Pytorch
    Run that 3D tensor through the trained model, which creates a prediction tensor
    Call functions to format that prediction tensor into a 1D binary numy array.
    Extract the CRS of one of the input visualization map and assign that CRS profile to the new predicted array.
    Write the georeferenced prediction to a geotiff file.

    Parameters
    ----------
    model : Pytorch model
        Pretrained model structure with loaded weights.
    filename : str
        Name of the weights file.
    tilenames : list
        List of filenames of tiles to run through the model.
    inputs_test_dim1 : str
        Name of the first visualization type.
    inputs_test_dim2 : str
        Name of the second visualization type.
    inputs_test_dim3 : str
        Name of the third visualization type.
    output_path : str
        Path to the CNN_output/Model_predictions folder that will hold the geotiffs created.

    Returns
    -------
    None.

    '''
    
    # Check if the output folder exists and create a new one if it doesn't
    isExist = os.path.exists(output_path)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(output_path)
       print("Created a new folder for the predictions")
    
    # Print statement to follow progress
    print(f"\nRunning {len(tilenames)} tiles through the model to create predictions. This may take a few minutes.\n")
    
    # Define keys of the dictionary to create the geodatabase
    keys = ["ids", "geometry", "score"]
    geo_dict = dict(zip(keys, ([] for _ in keys)))
    
    # Iterate through the list of filenames provided
    for file in tilenames:
        if file.endswith(".tif"):
            
            # Open the tiles with same name different band of same tile)
            image_dim1 = imread(os.path.join(inputs_test_dim1, file))
            image_dim2 = imread(os.path.join(inputs_test_dim2, file))
            image_dim3 = imread(os.path.join(inputs_test_dim3, file))
            
            # Preprocess the image to fit the trained format
            im = preprocess(image_dim1, image_dim2, image_dim3)
        
            # Load the image to the device where the model is loaded
            im = im.to(torch.device(device))
            
            with torch.no_grad():
                # Create the prediction of the mask using the trained model
                mask = model(im)
            
            # Prostprocess the mask (transform into a 1-dim numpy array)
            boxes, scores, mask = postprocess_MaskRCNN(mask)
        
            # georeference the predicted mask
            with rasterio.open(os.path.join(inputs_test_dim1, file)) as src:
                # Extract the profile of one of the vizualization tiles used as source
                profile = src.profile

                # Set nodata as 0
                profile.update(nodata = 0) 
                
                # Apply the profile extracted to the new image and write it to a new file.
                with rasterio.open(os.path.join(output_path, file), "w", **profile) as output_file:
                    output_file.write(mask, indexes = 1)

            # Add the bounding boxes as polygons with their prediction scores to the geo_dictionary
            geo_dict = export_bounding_boxes(geo_dict, boxes, scores, profile)
    
    # Assign to a geodatabase with the correct CRS
    polygons = gpd.GeoDataFrame(geo_dict, crs=profile['crs'])
    
    intersects_diss = merge_overlapping_bboxes(polygons)
    
    # Write as a shapefile
    intersects_diss.to_file(filename=rf"{output_path}_bbox.gpkg", driver="GPKG")
            
    return

def merge_tiles(pred_tiles_path, output_file):
    '''
    Merge all the predicted geotiffs into one big geotiff and then delete the individual geotiffs

    Parameters
    ----------
    pred_tiles_path : str
        Path to the predicted geotiff tiles.
    output_file : str
        Name of the big geotiff to create with the merged tiles.

    Returns
    -------
    None.

    '''
    
    # Get a list of all the raster files ending with ".tif" in the input folder
    file_list = [f for f in os.listdir(pred_tiles_path) if f.endswith('.tif')]
    
    # Create an empty list that will take on all merged tiles
    raster_to_mosaic = []
    
    # Loop through the raster files, open them, and append them to the list
    for p in file_list:
        raster = rasterio.open(os.path.join(pred_tiles_path, p))
        raster_to_mosaic.append(raster)
    
    # Merge them
    mosaic, output = merge(raster_to_mosaic)
    
    # Update the metadata of the merged raster
    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    })
    
    # Write the merged raster to the output file
    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    # Print statement to follow progress
    print("Merged raster file created: {}".format(output_file))
       
    return
    
def import_and_load_model_weights(device, backbone, weights_path, filename):
    '''
    Import the model structure and its weights
    Load them and send them to the appropriate device

    Parameters
    ----------
    device : str
        Name of the device ("cuda" or "cpu") depending on the computer.
    backbone : str
        Name of the backbone used to create the trained weights to import.
    weights_path : str
        Path to the folder that holds the weights files or to the .pt file that holds the model weights directly.
    filename : str
        Name of the weights file (if not already provided in weights_path). Can be None if it's included in weight_path.

    Returns
    -------
    model : Pytorch model
        Pytorch model structure with loaded weights and already sent to the proper device..

    '''
    
    # If a filename is provided (when called by main_with_metrics), use it to define the weights_path
    if filename != None:
        weights_path = rf'{weights_path}/{filename}.pt'
    
    # Import the correct model
    from mask_rcnn_backbones.mask_rcnn_resnet50 import build_model
    
    # Define the number of classes (should be 2 for background and one type of object)
    num_classes = 2
    
    # Upload the model from the mask_rcnn script
    model = build_model(num_classes)

    # Load the trained weights to the appropriate device and apply to the model
    if device == "cuda":
        checkpoint = torch.load(weights_path)
    else:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    # Send the model to the device (CUDA if on PC)
    model.to(device)
    
    # Return the loaded model sent to the device
    return model
 
def main_with_metrics(filename, cnn_output_path, data_path, separation_random, train_bounds, inputs_test, path_to_shp, threshold):
    '''
    Call most of the functions above to:
        Load the model and its weights, and send to the appropriate device
        Set the model to eval mode
        Iterate through the testing dataset and run them through the model to create predictions, which are exported as geotiff tiles
        Merge all predicted tiles into one big raster saved as a geotiff, vectorize it and save it as a shapefile
        Use the provided shapefile of annotated objects to calculate the object-per-object metrics using the provided threshold

    Parameters
    ----------
    filename : str
        Name of pretrained file (weights) that holds metadata.
    cnn_output_path : str
        Path to the CNN_output folder.
    data_path : str
        Path to the CNN_input folder that holds the visuzliation tiles' folders.
    separation_random : bool
        How the training/validation/testing datasets were separated. True for random, False for geographical.
    train_bounds : list
        Geographical bounds of the training dataset if separation_random is set to False.
    inputs_test : list
        List of the tile names that were set as testing dataset when running the model training (to keep the same).
    path_to_shp : str
        Path to the shapefile that holds the actual annotated objects to compare against the predictions.
    threshold : int
        All predicted polygons with area < that value will be deleted before calculating object-per-object metrics.

    Returns
    -------
    None.

    '''
    
    # Define some local variables
    # The path where the predicted tiles will be saved using the information we have here
    pred_tiles_path = rf'{cnn_output_path}/Model_predictions/{filename}'
    # The path and name of the merged raster to create
    path_to_ras =  rf'{pred_tiles_path}.tif'
    
    # Parse the filename to get the visualization maps names and backbone
    filename_list = filename.split("_")
    backbone = filename_list[1]
    vis1 = filename_list[7]
    vis2 = filename_list[8]
    vis3 = filename_list[9]
    im_size = filename_list[11]
    
    # Import and load the model and weights, and send to correct device
    model = import_and_load_model_weights(device, backbone, rf'{cnn_output_path}/Model_weights', filename)

    # Set the model to eval to avoid training it again.
    model.eval()
    
    # Add paths to datasets so the model knows where to look
    train_dir_dim1 = os.path.join(data_path, rf'Input_{vis1}_{im_size}')
    train_dir_dim2 = os.path.join(data_path, rf'Input_{vis2}_{im_size}')
    train_dir_dim3 = os.path.join(data_path, rf'Input_{vis3}_{im_size}')

    # Run each test tile through the model and create predictions, which are saved as geotiffs
    predict_on_new_tiles(model, filename, inputs_test, train_dir_dim1, train_dir_dim2, train_dir_dim3, pred_tiles_path)
    
    # Merge the predicted tiles to create one raster
    merge_tiles(pred_tiles_path, path_to_ras)
    
    # Prep the polygon dataset
    pred_poly = gpd.read_file(rf'{pred_tiles_path}_bbox.gpkg', driver="GPKG")
    
    # Import the testing polygon (filtered from the areas gdf)
    area_polys = gpd.read_file(rf'{cnn_output_path}/Model_predictions/{filename}_areas.gpkg')
    test_poly = area_polys[area_polys['data_type'] == "Testing"]
    
    # Calculate the metrics on the new predicted polygons
    TP, FN, FP, recall, precision, F1 = calculate_metrics.compute_object_metrics(path_to_shp, test_poly, pred_poly, threshold)
    
    logmet.populate_metrics(cnn_output_path, filename, "Post_threshold", threshold)
    logmet.populate_metrics(cnn_output_path, filename, "Object_recall", round(recall, 3))
    logmet.populate_metrics(cnn_output_path, filename, "Object_precision", round(precision, 3))
    logmet.populate_metrics(cnn_output_path, filename, "Object_F1", round(F1, 3))
    logmet.populate_metrics(cnn_output_path, filename, "Object_TP", TP)
    logmet.populate_metrics(cnn_output_path, filename, "Object_FN", FN)
    logmet.populate_metrics(cnn_output_path, filename, "Object_FP", FP)
    
    return

def main_without_metrics(path_to_weights_file, data_path):
    '''
    Call most of the functions above to:
        Load the model and its weights, and send to the appropriate device
        Set the model to eval mode
        Iterate through the provided dataset and run them through the model to create predictions, which are exported as geotiff tiles
        Merge all predicted tiles into one big raster saved as a geotiff, vectorize it and save it as a shapefile

    Parameters
    ----------
    path_to_weights_file : str
        Path to the weights file that will be used (pretrained model).
    data_path : str
        Path to the folder that holds the CNN_input folders.

    Returns
    -------
    path_to_ras : str
        Path to the raster created (if applicable). Used in the 03_predict_on_new_data.py script.

    '''
    
    # Define some local variables using the information provided
    # Get the filename from the path to weights
    path_sections = path_to_weights_file.split("/")
    weights_file = path_sections[-1]
    filename = weights_file.replace(".pt", "")
    # print(filename)
    
    # Parse the filename to get the visualization maps and backbone
    filename_list = filename.split("_")
    backbone = filename_list[1]
    vis1 = filename_list[7]
    vis2 = filename_list[8]
    vis3 = filename_list[9]
    im_size = filename_list[11]

    # Define the paths to the folder where we save cnn outputs and the name of the raster created
    cnn_output_path = rf'{data_path}/CNN_output'
    output_path = rf'{cnn_output_path}/Model_predictions/{filename}'
    path_to_ras = rf'{output_path}.tif'

    # Import and load the model and weights, and send to correct device
    model = import_and_load_model_weights(device, backbone, path_to_weights_file, filename = None)

    # Set the model to eval to avoid training it again.
    model.eval()

    # Get a list of all the tiles that will be run through the model
    list_tiles = os.listdir(rf'{data_path}/CNN_input/Input_{vis1}_{im_size}')

    # Add paths to datasets so the model knows where to look
    train_dir_dim1 = rf'{data_path}/CNN_input/Input_{vis1}_{im_size}'
    train_dir_dim2 = rf'{data_path}/CNN_input/Input_{vis2}_{im_size}'
    train_dir_dim3 = rf'{data_path}/CNN_input/Input_{vis3}_{im_size}'

    # Iterate through the tiles, run them through the model and predict the presence of objects
    predict_on_new_tiles(model, filename, list_tiles, train_dir_dim1, train_dir_dim2, train_dir_dim3, output_path)

    # Merge the predicted tiles to create one raster
    merge_tiles(output_path, path_to_ras)

    # Vectorize the predicted raster and save as shapefile
    calculate_metrics.vectorize(path_to_ras)
    
    return path_to_ras

# THE END

main_without_metrics('C:/Users/Claudine/Documents/CNN_models_data/Tar_kiln_RGB_test/CNN_output/Model_weights/MaskRCNN_ResNet50_5ep_0m_MAP_8bs_lrVariable_Slope_Slope_Slope_0Thresh_256_1741132454.pt', 
                     'C:/Users/Claudine/Documents/CNN_models_data/Tar_kiln_RGB_test')