#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Claudine Gravel-Miguel put this document together, but most of the functions were written by Katherine Peck and simplified by CGM
@date: 10 Jan 2024

@description: 
    This script contains functions to separate a raster into tiles that correspond to the squares of a provided shapefile grid

"""

# Reading and writing shapefiles, geodataframes
import geopandas as gpd

import rasterio
import rasterio.mask

import numpy as np

import math

import os

from skimage.io import imread

def format_grid(path_to_shp):
    '''
    This imports and formats a shapefile grid so it can be used by later code

    Parameters
    ----------
    path_to_shp : str
        Path to the shapefile grid created in QGIS or Python.

    Returns
    -------
    grid : GeoDataFrame
        Geopandas dataframe with the geometry of the grid (bottom left coordinates of each polygon).

    '''
    
    # Import the grid shapefile
    grid = gpd.read_file(path_to_shp)

    # Get the x and y anchors in named columns
    grid['x'] = grid['geometry'].bounds['minx']
    grid['y'] = grid['geometry'].bounds['miny']
    
    return(grid)
    

def scale_floats(array):
    '''
    Inputs for the ML process need to be in floats between 0-1
    This takes a numpy array and returns a 0-1 scaled version

    Parameters
    ----------
    array : NumPy array
        numpy array representing the raster image post-SLRM.

    Returns
    -------
    scaled : NumPy array
        numpy array of floats with scaled values.

    '''
    
    # Min value in the array (ignoring NAs)
    array_min = np.nanmin(array)
    
    # Moving all values so that the minimum is 0
    array_to_zero = array - array_min
    
    # New max value in the array (ignoring NAs). Replacing its value with a tiny value if it is 0.
    array_max = np.nanmax(array_to_zero)
    if array_max == 0:
        array_max = 0.00001
    
    # Divide the array values by its max to constraint values to 0-1
    scaled = array_to_zero/array_max
    
    return scaled


def tile_from_grid(bigraster, path_to_grid, outfolder, tile_size, is_mask, drop_nan):
    '''
    Extract rasters by the mask of the grid and save as GeoTiffs

    Parameters
    ----------
    bigraster : str
        filepath of raster to be tiled.
    path_to_grid : GeoDataFrame
        geopandas gpd of output tile shapes.
    outfolder : str
        filepath for output folder.
    tile_size : int
        desired tile height and width.
    is_mask : bool
        If the raster tiled is a mask (in which case, it does not get rescaled to 0-1).
    drop_nan : bool
        If we want to drop any tile with nan value or keep them but transform nan to 0.

    Raises
    ------
    Exception
        Raise an exception if there is a disconnect between the tile_size provided and the resulting tiles' size (likely due to resolution).

    Returns
    -------
    None.

    '''
    
    # Check if the outfolder already exists, and create a new one if it does not.
    isExist = os.path.exists(outfolder)
    if isExist:
        print("The folder for the tiles already exists. New tiles will be added to it.")
        
    if not isExist:
        # Create a new folder because it does not exist
        os.makedirs(outfolder)
        print("Created a new folder for the tiles")
    
    # Import and format the grid to fit requirements of code below
    grid = format_grid(path_to_grid)
    
    # Create a counter to catch resolution errors
    counter = 0
    
    # Load the bigraster (only once)
    with rasterio.open(bigraster) as src:
        
        # Iterate through each square in the provided grid
        for index, row in grid.iterrows():
            out_image, out_transform = rasterio.mask.mask(src, [row['geometry']], crop = True)
            out_meta = src.meta
            
            # If the created tile is not square (was at the edge of the raster map), it is ignored
            if out_image.shape[1] != tile_size or out_image.shape[2] != tile_size:
                # Update the counter to print an error message later on.
                counter += 1
                #print(f"Tile_{str(math.floor(row['x']))}_{str(math.floor(row['y']))} goes beyond the raster and has shape {out_image.shape}. Therefore, it will be ignored.")
                pass
            
            else:
                # Height and width are the same dimensions as the desired tile size
                out_meta.update({"driver": "GTiff",
                             "height": tile_size,
                             "width": tile_size,
                             "transform": out_transform})
                
                # Ignores tiles with NAs if drop_nan is set to True
                if drop_nan == True and np.count_nonzero(out_image < -9000) > 0:
                    print(f"Dropping Tile{int(row['x'])}_{int(row['y'])} because it has NAs")
                    pass
                
                else:
                    # Filename should reference the lower left point of the tile
                    filename = outfolder + "/Tile" + str(math.floor(row['x'])) + "_" + str(math.floor(row['y'])) + ".tif"
                    
                    # -999999 or -9999 are the NoData value for these rasters - this transforms NAs into 0s before rescaling or cutting
                    out_image[out_image < -9000] = 0
                    
                    # Scale float values between 0 - 1 if the raster is NOT a mask
                    if is_mask == True:
                        pass
                    else:
                        out_image = scale_floats(out_image)
                                            
                    # Export the tile as a geotiff
                    with rasterio.open(filename, "w", **out_meta) as dest:
                        dest.write(out_image)
    
    # If all tiles were ignored, it means there is a problem in the parameter values provided or the resolution of the map.
    if counter == len(grid):
        raise Exception("The size of the tiles would be different from the value provided in tile_size, so they were all ignored. Check your large_raster's resolution, the tile_size value you provided, and how those fit with the size of the grid squares")

    return
        
def check_tile_size(outfolder, tile_size):
    '''
    Check the size of the tiles created against the value entered

    Parameters
    ----------
    outfolder : str
        Path to the folder that holds the created tiles.
    tile_size : int
        Size wanted for the tile.

    Returns
    -------
    None.

    '''
    
    # Create a list of the tiles' names
    filelist = os.listdir(outfolder)

    # Load the first tile (this is done only on one tile because they would already show a problem) and calculate its size
    m = imread(f'{outfolder}/{filelist[1]}')
    m_size = m.shape[1]
    
    # Print the size of the created tile
    print(f'The tiles created are {m_size}x{m_size}')
    
    # If the size of the tile is the same as a the provided tile_size, nothing happens because everything is OK.
    if m_size == tile_size:
        pass
    # If the size is different, it prints a warning statement
    else:
        print('WARNING: The size of the tiles you created differ from the tile_size entered above. Double-check your paths and values entered.')

# THE END