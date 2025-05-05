# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 15:09:39 2024

@author: Claudine Gravel-Miguel

@description: This script cleans the separated training/validation/testing datasets by removing overlapping tiles from one dataset, when applicable.
It then creates a polygon covering the merged footprints of the tiles included in each dataset (training, validation, and testing) and saves it as a gpkg

This script is called by the maskrcnn_main script after calling separate_datasets

"""

import re
import random
import geopandas as gpd
import rasterio
from rasterio.features import dataset_features
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

def calculate_min_distance(file):
    '''
    Calculates the minimum distance allowed between tiles in different datasets (training vs val vs test) to avoid overlap.
    The distance is based on the size of the tiles and their resolution.

    Parameters
    ----------
    file : str
        Path to a visualization raster tile.

    Returns
    -------
    min_distance : float
        Minimum distance allowed between tiles from different datasets.

    '''
    
    with rasterio.open(file) as src:
        # Extract features from the raster
        profile = src.profile
        res_x = profile['transform'][0]
        im_size = profile['width']
        
        # Calculate the minimum distance allowed between two tiles' lower left corners
        min_distance = res_x * im_size
        
        return min_distance

def extract_coordinates(filename):
    '''
    Extract the x and y coordinates from a filename like 'TileX_Y.tif'.

    Parameters
    ----------
    filename : str
        Tile names extracted from the lists created for training, validation, and testing datasets.

    Returns
    -------
    tuple of XY coordinates
        XY coordinates extracted from the filename string, which represents that tile's lower left corner's coordinate.

    '''
    
    match = re.match(r"Tile(\d+)_(\d+)\.tif", filename)
    if match:
        return int(match.group(1)), int(match.group(2))  # x, y as integers
    return None

def clean_overlapping_tiles(filenames1, filenames2, min_distance):
    print("Removing overlapping tiles. This may take a few seconds.")
    
    # Initialize coordinate datasets
    coords1 = np.array([extract_coordinates(filename) for filename in filenames1])
    coords2 = np.array([extract_coordinates(filename) for filename in filenames2])

    # Determine the ratio to delete more from the bigger dataset to preserve the original ratio as much as possible.
    ratio1 = len(coords1)/(len(coords1)+len(coords2))

    # Build KDTree for coords2
    tree = cKDTree(coords2)
    
    # Initialize lists for filtered filenames and coordinates
    filtered_filenames1 = filenames1[:]
    filtered_filenames2 = filenames2[:]
     
    # Iterate through coords1 and check distances to coords2
    i = 0
    counter = 0
    while i < len(filtered_filenames1):
        point = coords1[i]
         
        # Query for the nearest neighbor in coords2 within min_distance
        distances, indices = tree.query(point, k=1, distance_upper_bound=min_distance)
         
        if distances != np.inf:  # Found a neighbor within min_distance
            counter += 1
            if random.random() < ratio1:
                # Remove from filenames1 and coords1
                del filtered_filenames1[i]
                coords1 = np.delete(coords1, i, axis=0)
            else:
                # Remove from filenames2 and coords2
                del filtered_filenames2[indices]
                coords2 = np.delete(coords2, indices, axis=0)
                # Rebuild KDTree after modifying coords2
                tree = cKDTree(coords2)
             
            # Skip incrementing `i` if we removed from filenames1
            continue
         
        # Move to the next point
        i += 1
    
    print(f"Removed {counter} overlapping tiles.")
    
    return filtered_filenames1, filtered_filenames2

def create_poly_from_tiles(list_tiles, data_type):
    '''
    Create a polygon(s) that include all the tiles used in one dataset (either training, validation, or testing).

    Parameters
    ----------
    list_tiles : list
        List of tile names included in the dataset to vectorize.

    Returns
    -------
    None.

    '''
    # Create the geodataframe from the first tile
    with rasterio.open(list_tiles[0]) as src:
        # Extract features from the raster
        gdf = gpd.GeoDataFrame.from_features(dataset_features(src, bidx=1, as_mask=True, geographic=False, band=False))
    
    for tile in list_tiles[1:]:
        with rasterio.open(tile) as src:
            # Extract features from the raster
            features = list(dataset_features(src, bidx=1, as_mask=True, geographic=False, band=False))
            
            if features: # Check if features are returned
                gdf_tile = gpd.GeoDataFrame.from_features(features)
                gdf = pd.concat([gdf, gdf_tile], ignore_index=True)
            else:
                print(f"No features extracted for {tile}")
    
    # Drop unused field
    gdf = gdf.drop(columns = ['val'])
    
    # Set the CRS from the first tiles
    with rasterio.open(list_tiles[0]) as src:
        try:
            gdf.set_crs(src.crs, inplace=True)
        except:
            print(f"Tile {list_tiles[0]} has no CRS!")
        
    # Create a simplified geodatabase (single polygons instead of overlapping grids)
    gdf_polygons = gdf.dissolve().explode(index_parts = True)
    # Drop columns that no longer serve us
    gdf_polygons = gdf_polygons.drop(columns = ['filename'])
    # Create a new column that takes the dataset type
    gdf_polygons['data_type'] = data_type
                
    return(gdf_polygons)

# THE END