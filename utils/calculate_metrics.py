"""
Created on Wed Dec 20 2023

@authors: Katherine Peck and Claudine Gravel-Miguel

@description: This script holds functions that vectorize the predicted presence pixels, 
do some post-processing cleaning by removing polygons that are smaller than a provided threshold,
and calculate the standard metrics (TP, TN, FP, FN, recall, precision, F1 score, and MCC) by comparing the cleaned predictions to a provided shapefile of annotated features.

"""

# Import necessary modules

import geopandas as gpd

import rasterio
from rasterio import features

from shapely.geometry import shape

import numpy as np

def vectorize(rasterpath):
    '''
    Given a binary raster, return a geodataframe of polygons for every pixel with a value of 1

    Parameters
    ----------
    rasterpath : str
        binary raster filepath.

    Returns
    -------
    features_gdf : GeoDataFrame
        geopandas gdf of polygons.

    '''
    
    # Open raster 
    with rasterio.open(rasterpath) as src:
        data = np.float32(src.read())
        data[np.isnan(data)] = src.nodata
        transform = src.transform
    
    # Create feature mask 
    mask = data > 0
    
    # Get shapes and values based on the mask
    # Create GeoJSON-like object
    shapeDict = features.shapes(data, mask = mask, transform = transform)
    feats = []
    geoms = []
    
    # Append shapes to empty lists
    for key, value in shapeDict:
        feats.append(value)
        geoms.append(shape(key))
    crs = src.crs
    
    # Create new geodatarame from lists with original raster CRS
    features_gdf = gpd.GeoDataFrame({'feats': feats, 'geometry': geoms}, crs = crs)
    
    # Test exporting the result as a shapefile as well
    shp_filepath = rasterpath.replace(".tif", ".gpkg")
    
    # Add area
    features_for_shp = features_gdf
    features_for_shp["area"] = features_for_shp.area
    features_for_shp.to_file(shp_filepath, driver='GPKG')
    
    # Return geodataframe
    return features_gdf

# Define the main helper function, which also calls vectorize() (defined above)
def compute_object_metrics(path_to_shp, test_poly, pred_poly, threshold):
    '''
    Given a shapefile of actual features, a raster of predicted features,
    and an integer threshold for feature size, prints the recall, precision, and F1 values.

    Parameters
    ----------
    path_to_shp : str
        filepath for shapefile of actual features.
    pred_poly : GeoPandas vector
        Imported shapefile of predicted polygons (can be bounding boxes for Mask and Faster RCNN).
    threshold : int
        integer for max size of features to be removed from the predicted features.

    Returns
    -------
    TP : int
        Number of true positives (annotations correctily predicted by the model)
    FN : int
        Number of False negatives (annotations not predicted by the model)
    FP : int
        Numer of False positives (predictions that do not represent true presence)
    recall : float
        The percentage of annotations correctly predicted by the model
    precision : float
        The percentage of model predictions that are correct
    f1 : float
        The harmonic mean of recall and precision

    '''

    # Read in shapefile of predicted features
    actual = gpd.read_file(path_to_shp)
    
    # Clip the annotated vectors
    if test_poly is not None:
        clipped_actual = gpd.clip(actual, test_poly)
        actual = clipped_actual

    # Filter out the smaller polygons using input area threshold
    pred_poly = pred_poly[pred_poly.area > threshold]

    # Calculate FN (false negative), TP (true positive), and FP (false positive)
    TP = 0
    FN = 0
    for actual_geom in actual.geometry:
        
        # This catches actual polygons with faulty geometries and ignores them instead of throwing an error.
        if actual_geom != None:
            
            # If a shape in the predicted file intersects a shape in the actual file, it's a true positive
            # Otherwise, it's a true negative
            overlap = any(actual_geom.intersects(pred_geom) for pred_geom in pred_poly.geometry)
            if overlap:
                TP += 1
            else:
                FN += 1
        else:
            print(actual_geom)

    print("TP: " + str(TP))
    print("FN: " + str(FN))

    FP = 0
    
    # For all shapes in the gdf of predicted polygons
    for pred_geom in pred_poly.geometry:
        # If a polygon does not intersect a feature in the actual shapefile, it is a false positive
        overlap = any(pred_geom.intersects(actual_geom) for actual_geom in actual.geometry)
        if not overlap:
            FP += 1

    print("FP: " + str(FP))

    # CALCULATE AND PRINT METRICS
    recall = TP / (TP + FN + 0.00001)
    precision = TP / (TP + FP + 0.00001)
    F1 = (2 * recall * precision) / (recall + precision + 0.00001)

    print("Recall:", round(recall, 3))
    print("Precision:", round(precision, 3))
    print("F1:", round(F1, 3))
    
    return TP, FN, FP, recall, precision, F1

# Define an alternate helper function, which saves metrics to a list
def save_metrics(path_to_shp, path_to_ras, threshold):
    '''
    Given a shapefile of actual features, a raster of predicted features,
    and an integer threshold for feature size, returns the recall, precision, F1- score and MCC values.

    Parameters
    ----------
    path_to_shp : str
        filepath for shapefile of actual features.
    path_to_ras : str
        filepath for raster of detected features.
    threshold : int
        integer for max size of features to be removed from the predicted features.

    Returns
    -------
    metrics : list
        list of metrics in the order [recall, precision, f1, MCC].

    '''
    # Read in shapefile of predicted features
    actual = gpd.read_file(path_to_shp)
    print("Read the Shapefile")
    
    pred_poly = vectorize(path_to_ras)
    print("Vectorized the raster")

    # Filter out the smaller polygons
    # These features are in UTM, so area calculation should be in m^2 
    # at the moment, I don't think we need to use pint to convert, but it depends on the inputs from the previous step
    pred_poly = pred_poly[pred_poly.area > threshold]
    print("Filtered the polygons")

    # Calculate FN (false negative), TP (true positive), and FP (false positive)
    TP = 0
    FN = 0
    for actual_geom in actual.geometry:
        # If a shape in the predicted file intersects a shape in the actual file, it's a true positive
        # Otherwise, it's a true negative
        overlap = any(actual_geom.intersects(pred_geom) for pred_geom in pred_poly.geometry)
        if overlap:
            TP += 1
        else:
            FN += 1

    print(f"TP: {TP}, and FN: {FN}")
    FP = 0
    
    # For all shapes in the gdf of predicted polygons
    for pred_geom in pred_poly.geometry:
        # If a polygon does not intersect a feature in the actual shapefile, it is a false positive
        overlap = any(pred_geom.intersects(actual_geom) for actual_geom in actual.geometry)
        if not overlap:
            FP += 1

    print(f"FP: {FP}")
    
    # CALCULATE AND PRINT METRICS
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * recall * precision) / (recall + precision)

    metrics = [recall, precision, F1]
    
    return metrics

def calculate_mcc(merged_ras):
    '''
    This takes a numpy array with specific values associated with TP (12), TN (6), FP (7), and FN (11), and calculates the basic metrics and MCC on pixels.

    Parameters
    ----------
    merged_ras : array
        The merged raster of annotated and predicted features, where values from the confusion matric (TP, FP, TN, FN) have specific values.

    Returns
    -------
    None.

    '''
    
    # Calculate the confusion matrix
    TP = sum(merged_ras[merged_ras == 12])
    TN = sum(merged_ras[merged_ras == 6])
    FP = sum(merged_ras[merged_ras == 7])
    FN = sum(merged_ras[merged_ras == 11])
    
    # Calculate all standard metrics
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    F1 = (2 * recall * precision) / (recall + precision)
    MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    
    print("These values are calculated on ALL raster pixels of the given maps. If the rasters had a lot of NA values around, those might not be representative.")
    print(f"TP: {TP}")
    print(f"FP: {FP}")
    print(f"TN: {TN}")
    print(f"FN: {FN}")
    print(f"Recall: {round(recall, 3)}, Precision: {round(precision, 3)}, F1: {round(F1, 3)}, and MCC: {round(MCC, 3)}")
    
    return

# THE END
    