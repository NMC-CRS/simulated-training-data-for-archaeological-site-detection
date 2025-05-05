#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:36:55 2024

@author: Claudine Gravel-Miguel put this document together, but the create_tile_bounds function was written by Katherine Peck

This holds functions that are used to create a grid that covers the area of a provided raster, with squares of a specified size, as well as to combine overlapping grids and save as a shapefile.

"""

import rasterio
import rasterio.mask

from shapely.geometry import box
from shapely import Polygon

import geopandas as gpd

import numpy as np

import pandas as pd

def create_tile_bounds(largeraster, tilewidth, tileheight, xoffset = None, yoffset = None):
    '''
    Given a GeoTIFF, return a geodataframe of a grid covering the entire extent with tiles of a user-specified size

    Parameters
    ----------
    largeraster : str
        GeoTIFF file path.
    tilewidth : int
        Integer of desired tile width.
    tileheight : int
        Integer of desired tile height.
    xoffset : int, optional
        Integer of desired x (easting) offset from origin. The default is None.
    yoffset : int, optional
        Integer of desired y (northin) offset from origin. The default is None.

    Returns
    -------
    grid : GeoDataFrame
        Geopandas GDF of grid.

    '''
    
    # Load the raster and get some metadata from it
    img = rasterio.open(largeraster)
    bound = img.bounds
    geoms = box(*bound)
    boundingBox = gpd.GeoDataFrame({'id':1, 'geometry':[geoms]}, crs = img.crs)
    
    # This section modified from:
    # https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
    xmin, ymin, xmax, ymax = boundingBox.total_bounds
    
    # Transforms the input dimensions into number of pixels
    pixel_x = img.transform[0]
    pixel_y = -img.transform[4]
    length = tileheight * pixel_y
    wide = tilewidth * pixel_x
    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))
    
    # If offset values are provided, add these to each column/row value
    if xoffset is not None:
        cols = [x + float(xoffset * pixel_x) for x in cols]
    else:
        pass
    
    if yoffset is not None:
        rows = [y + float(yoffset * pixel_y) for y in rows]
    else:
        pass
    
    # Use these columns and rows to create Shapely polygons
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x+wide, y), (x+wide, y+length), (x, y+length)]))
    grid = gpd.GeoDataFrame({'geometry':polygons}, crs = img.crs)
    grid['x'] = grid['geometry'].bounds['minx']
    grid['y'] = grid['geometry'].bounds['miny']
    
    return grid

def join_and_save_overlapping_grids(grid_1, grid_2, path_out_shp):
    '''
    Join two overlapping grids into one objects that is exported as a shapefile.

    Parameters
    ----------
    grid_1 : GeoDataFrame
        Geopandas dataframe of the first grid.
    grid_2 : GeoDataFrame
        Geopandas dataframe of the second (overlapping) grid.
    path_out_shp : str
        Path of the file that will be created (shapefile).

    Returns
    -------
    None.

    '''
    
    # Join the two grids into one
    joined = pd.concat([grid_1.geometry, grid_2.geometry])

    # Export to a shapefile
    joined.to_file(path_out_shp, driver='ESRI Shapefile')
    
    return

# THE END