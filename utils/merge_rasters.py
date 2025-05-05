#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:52:12 2024

@authors: Claudine Gravel-Miguel and Katherine Peck

@description: This script takes in a raster of predictions (from polygons cleaned in QGIS) and the raster of annotations and creates a map where TP, TN, FP, and FN have different values that can be easily visualized in QGIS

"""

# setup
import rasterio
import rasterio.plot

def set_merged_raster_values(predicted_raster_path, annotated_raster_path, output_path):
    '''
    This opens the two rasters, checks for similar bounds.
    If the maps overlap perfectly, this reclassifies their values, merges the two maps, calculate metrics, and exports the merged raster as a geotiff

    Parameters
    ----------
    predicted_raster_path : str
        Path to the raster that holds predicted features (should be binary with 0 or -999999 = absence and 1 = presence).
    annotated_raster_path : str
        Path to the raster that holds the annotations (binary -999999(NA)-1).
    output_path : str
        The path to the merged geotiff created.

    Raises
    ------
    Exception
        If the two rasters do not have the same bounds (extent, resolution), adding their data would create errors, so this stops the whole script.

    Returns
    -------
    merged_array : numpy array
        Summed reclassified values. This is used by the 05_ script to call the relevant metrics function.

    '''
    
    with rasterio.open(predicted_raster_path) as pred_ras:
        
        # Extract the profile of the predicted map to use for the new geotiff
        profile = pred_ras.profile

        # Read the data as array
        pred = pred_ras.read(1)

        with rasterio.open(annotated_raster_path) as actual_ras:
            
            # Read the data as array
            actual = actual_ras.read(1)
            
            # Check for identical bounds 
            if pred_ras.bounds != actual_ras.bounds:
                raise Exception("IMPORTANT: The two maps do not have the same bounds, so the metrics are not calculated, as they would be inaccurate")
            
            # Reclassify the annotations
            actual[actual < -9000] = 5
            actual[actual == 1] = 10
            
            # Reclassify the predictions
            pred[pred == 1] = 2
            pred[pred < -9000] = 1
            
            # Sum the two arrays
            merged_array = actual + pred

            # Apply the profile extracted to a new empty geotiff and write the merged array values to it.
            with rasterio.open(output_path, "w", **profile) as output_file:
                output_file.write(merged_array, indexes = 1)
    
    return merged_array

# THE END
