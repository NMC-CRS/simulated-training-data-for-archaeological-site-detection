import rasterio
import rasterio.features
import geopandas as gpd
import pandas as pd
import numpy as np
import random
import math
import shapely
from shapely.geometry import Point, Polygon, box

#slope() and noisy_poly() are called by the main functions create_tar_kiln_like_objects(), which corresponds to Method 1 and create_simple_objects() which corresponds to Method 2

#Get raster slope
def slope(array:np.ndarray) -> np.ndarray:
    """
    Returns the slope of an input DEM 

    Parameters
    ----------
    array : np.ndarray
        DEM raster, opened and read as a numpy array

    Returns
    -------
    np.ndarray
        Numpy array with slope calculated in percent (decimal)
    """
    #References this StackExchange post:
    ##https://gis.stackexchange.com/questions/361837/calculating-slope-of-numpy-array-using-gdal-demprocessing
    px, py = np.gradient(array)
    array_slope = np.sqrt(px ** 2 + py ** 2)
    return array_slope

def noisy_poly(poly:shapely.Polygon) -> shapely.Polygon:
    """
    Add Gaussian noise to the exterior of an input polygon

    Parameters
    ----------
    poly : shapely.Polygon
        Shapely Polygon to which noise will be added (e.g. Polygon([(0, 0), (1, 1), (1, 1)]))

    Returns
    -------
    shapely.Polygon
        Polygon with noisy edges
    """
    xx, yy = poly.exterior.coords.xy
    x = xx.tolist()
    y = yy.tolist()
    clean_signal = pd.DataFrame({'x': x, 'y': y})
    mu, sigma = 0, 0.25
    noise = np.random.normal(mu, sigma, [len(clean_signal['x']), 2])
    noisy = clean_signal + noise
    x = noisy['x'].tolist()
    y = noisy['y'].tolist()
    coords = [(x, y) for x, y in zip(x, y)]
    noisy_polygon = Polygon(coords)
    return noisy_polygon

#Method 1
    
def create_tar_kiln_like_objects(raster_path:str, num_features:int, streams_to_exclude:str, stream_buffer:int, feature_radius:int, slope_threshold:float)->list:
    """Adds a specified number of possible targets to an input raster

    Parameters
    ----------
    raster_path : str
        File path of raster to which simulated features will be added
    num_features : int
        Number of features that will be added to the raster
    streams_to_exclude : str
        File path of shapefile of streams to exclude from feature placement
    stream_buffer : int
        Buffer distance around input stream shapefile
    feature_radius : int
        Final feature radius
    slope_threshold : float
        Maximum slope percent, expressed as a decimal, on which features will be placed

    Returns
    -------
    list
        Format: [annotation mask (binary), annotation mask (RGB), modified raster, annotation GeoDataFrame]
        The annotation masks and modified raster are numpy arrays and can be saved as georeferenced rasters using rasterio
    """

    testimage = rasterio.open(raster_path)
    testimage_read = rasterio.open(raster_path).read(1)
    #Read in drainages file
    streams = gpd.read_file(streams_to_exclude)
    buff = streams.buffer(stream_buffer)
    buff = gpd.GeoDataFrame(geometry = buff, crs = streams.crs)
    #Provides an integer value for rasterizing the vector buffer
    buff['val'] = 2
    #Rasterize drainage exclusion
    with testimage as src:
        feats = rasterio.features.rasterize(shapes = ((geom, value) for geom, value in zip(buff.geometry, buff['val'])),
                                            out_shape = src.shape,
                                            transform = src.transform,
                                            fill = 0)
    #Get image bounds as a shapely box
    bounds = testimage.bounds
    geom = box(*bounds)
    #Turn image bounds into coordinates
    min_x, min_y, max_x, max_y = geom.bounds
    im_slope = slope(testimage_read)
    x, y = np.where(im_slope < slope_threshold)
    exclusion = np.zeros((testimage.shape[0], testimage.shape[1]))
    pixelSizeX = testimage.transform[0]
    pixelSizeY =-testimage.transform[4]
    for x, y in zip(x, y):
        exclusion[int(round(x*pixelSizeX)), int(round(y*pixelSizeY))] = 1 #This should make the function resolution independent
    x, y = np.where(feats + exclusion == 1)
    final_exclusion = np.zeros((testimage.shape[0], testimage.shape[1]))
    for x, y in zip(x, y):
        final_exclusion[x, y] = 1
    lowslope = np.where(final_exclusion == 1)
    xlist = lowslope[1].tolist()
    ylist = lowslope[0].tolist()
    redo = True
    #The code in this while loop drops any features whose exteriors would overlap after feature creation
    #This also means that of the n random coordinates, some will not be turned into features
    #Depending on where the randomly selected points occur, the number of "dropped" tar kilns will vary
    #The loop condition is set to True by default
    #After creating all the features and dropping overlapping features, the loop condition is set to False only if the number of features == input number of features
    #If you need your output to include an EXACT number of features, you should keep this while loop
    #If not, consider removing it - for a very large num_features, it can take a long time to reach the condition
    #In a very small raster, it might not be possible to ever reach the target number
    while redo:    
        coord_list = []
        for x, y in zip(xlist, ylist):
            coord_list.append([x + min_x, y + min_y])
        coord_list = random.sample(coord_list, num_features)
        #Need to make sure none of these tar kilns are touching each other
        test_gdf = gpd.GeoDataFrame(geometry = [Point(x) for x in coord_list], crs = testimage.crs)
        buffer_test = gpd.GeoDataFrame(geometry = test_gdf.buffer(feature_radius + 5), crs = testimage.crs)
        #Reference
        #https://gis.stackexchange.com/questions/457457/selecting-and-removing-polygons-that-overlap-within-a-single-geodataframe-in-pyt
        #Drop any features that overlap with the final exterior
        diss = buffer_test.dissolve()
        singleparts = gpd.GeoDataFrame(geometry = diss.apply(lambda x: [part for part in x.geometry.geoms], axis = 1).explode(), crs = diss.crs)
        singleparts["cluster"] = range(singleparts.shape[0])
        joined = gpd.sjoin(buffer_test, singleparts)
        joined = joined.merge(joined.groupby("cluster")["cluster"].count().rename("polycount"), left_on = 'cluster', right_index = True)
        joined = joined.loc[joined["polycount"] == 1]
        #Turn these buffered shapes back into points
        #Now the features should not overlap
        joined['centroid'] = joined['geometry'].centroid
        coord_list = [[x.x, x.y] for x in joined['centroid']]
        if len(coord_list) == num_features:
            redo = False
        else:
            redo = True
    #Create features
    all_feats = []
    counter = 0
    for rand_coord in coord_list:
        circ_features = []
        new_feature_radius = feature_radius + random.uniform(-2, .5)
        #Outer berm
        #Add some noise in a Gaussian distribution to the exterior of the shape
        polygon = shapely.buffer(Point(rand_coord), new_feature_radius)
        xx, yy = polygon.exterior.coords.xy
        x = xx.tolist()
        y = yy.tolist()
        clean_signal = pd.DataFrame({'x': x, 'y': y})
        mu, sigma = 0, 0.25
        noise = np.random.normal(mu, sigma, [len(clean_signal['x']), 2])
        noisy = clean_signal + noise
        x = noisy['x'].tolist()
        y = noisy['y'].tolist()
        coords = [(x, y) for x, y in zip(x, y)]
        noisy_circle = Polygon(coords)
        #Append exterior
        circ_features.append(noisy_circle)
        #Append interior
        berm_width = random.uniform(2.5, 4)
        intr = shapely.buffer(Point(rand_coord), new_feature_radius - berm_width)
        circ_features.append(intr)
        #Collection pit
        theta = random.randrange(225,315)
        offset_distance = random.uniform(0, 1)
        edge_x = rand_coord[0] + (new_feature_radius + 2 + offset_distance) * math.sin(math.radians(theta))
        edge_y = rand_coord[1] + (new_feature_radius + 2 + offset_distance) * math.cos(math.radians(theta))
        size = (new_feature_radius * 2) / 4
        left = Point([edge_x - size, edge_y + size])
        bottom = Point([edge_x - size, edge_y])
        right = Point([edge_x, edge_y])
        top = Point([edge_x, edge_y + size])
        #Turn these points into a polygon
        new_poly = Polygon([left, bottom, right, top])
        pit = shapely.affinity.rotate(new_poly, 270 - theta, 'center')
        pit_ext = shapely.buffer(pit, .5)
        circ_features.append(pit_ext)
        gdf = gpd.GeoDataFrame(geometry = circ_features)
        all_feats.append(gdf)
        counter += 1
    for_mask = []
    for_mask_pit = []
    for gdf in all_feats:
        #Append largest feature circle
        for_mask.append(gdf.iloc[0]['geometry'])
        #Append shape of collection pit
        for_mask_pit.append(gdf.iloc[-1]['geometry'])
        with rasterio.open(raster_path) as src:
            ditch = rasterio.features.rasterize(
                [gdf.iloc[0]['geometry']], 
                out_shape=src.shape,
                transform=src.transform)
        with rasterio.open(raster_path) as src:
            below_grade = rasterio.features.rasterize(
            [gdf.iloc[1]['geometry']],
            out_shape = src.shape,
            transform = src.transform)
        with rasterio.open(raster_path) as src:
            pit_exterior = rasterio.features.rasterize(
                [gdf.iloc[2]['geometry']],
                out_shape = src.shape,
                transform = src.transform)
        #Get coordinates of all pixels in the image that compose the berm
        x_indices, y_indices = np.where(ditch == 1)
        #Turn those coordinates into lists
        x_indices = x_indices.tolist()
        y_indices = y_indices.tolist()
        #Iterate through lists and replace the same coordinate in the original raster with the ditch
        for x, y in zip(x_indices, y_indices):
            value = testimage_read[x, y] + 1 #Adding elevation in raster units
            #replace that value in the raster
            testimage_read[x, y] = value 
        #Get coordinates of all pixels in the image that compose the center
        x_indices, y_indices = np.where(below_grade == 1)
        #Turn those coordinates into lists
        x_indices = x_indices.tolist()
        y_indices = y_indices.tolist()
        #Iterate through lists and replace the same coordinate in the original raster with the ditch
        for x, y in zip(x_indices, y_indices):
            value = testimage_read[x, y] - .75 #the higher this value, the "lower" the structure
            #Replace that value in the raster
            testimage_read[x, y] = value 
        #Get coordinates of all pixels in the image that compose the collection pit center
        x_indices, y_indices = np.where(pit_exterior == 1)
        x_indices = x_indices.tolist()
        y_indices = y_indices.tolist()
        elevations = []
        for x, y in zip(x_indices, y_indices):
            elevations.append(testimage_read[x, y])
        if len(elevations) < 1:
            max_elev = testimage_read.max(0)
        else:
            max_elev = max(elevations)
        #Iterate through lists and replace the same coordinate in the original raster with the rectangle
        for x, y in zip(x_indices, y_indices):
            height = max_elev
            read = testimage_read[x, y]
            value = height - read
            new_value = read + value - 1
            #replace that value in the raster
            testimage_read[x, y] = new_value 
    #Create annotation mask (circle)
    buffer_gdf = gpd.GeoDataFrame(geometry = shapely.buffer(for_mask, 1), crs = testimage.crs)
    #Create annotation mask (pit)
    #To ensure that these will be one single file, the shapes need a small buffer around them
    buffer_gdf_pit = gpd.GeoDataFrame(geometry = shapely.buffer(for_mask_pit, 2.2), crs = testimage.crs)
    #Create merged file
    shape_union = buffer_gdf.overlay(buffer_gdf_pit, how = 'union')
    shape_union = shape_union.dissolve()
    shape_union = shape_union.explode(index_parts = True)
    #Clip shapes by bounding box of raster
    bounds = testimage.bounds
    geom = box(*bounds)
    raster_bounds = gpd.GeoDataFrame(geometry = [geom], crs = testimage.crs)
    shape_union = gpd.overlay(shape_union, raster_bounds, how = 'intersection')
    with rasterio.open(raster_path) as src:
        annotation_mask = rasterio.features.rasterize(
            shape_union['geometry'], 
            out_shape=src.shape,
            transform=src.transform)
    shape_union['val'] = range(1, len(shape_union.index) + 1)
    with rasterio.open(raster_path) as src:
        annotation_mask_RGB = rasterio.features.rasterize(
            shapes = ((geom,value) for geom, value in zip(shape_union.geometry, shape_union.val)), 
            out_shape=src.shape,
            transform=src.transform)
    return [annotation_mask, annotation_mask_RGB, testimage_read, shape_union]

#Method 2

def create_simple_objects(raster:str, no_points:int, min_radius:float, max_radius:float, streams_vect:str, stream_buff:int, slope_threshold:float) -> list:
    """
    Add simple objects (circular berms, no pit) to an existing raster

    Parameters
    ----------
    raster : str
        file path to DEM to which simulated features will be added
    no_points : int
        number of features to add to raster
    min_radius : float
        minimum feature radius
    max_radius : float
        maximum feature radius
    streams_vect : str
        file path to vector file (e.g., ESRI shapefile, GeoPackage) with streams to exclude 
    stream_buff : int
        desired buffer area to exclude around drainage vector
    slope_threshold : float
        slope threshold for exclusion model (maximum slope on which objects can be placed)

    Returns
    -------
    list
        Format: [annotation mask (binary), annotation mask (RGB), modified raster, annotation GeoDataFrame]
        The annotation masks and modified raster are numpy arrays and can be saved as georeferenced rasters using rasterio
    """
    read_image = rasterio.open(raster).read(1)
    open_image = rasterio.open(raster)
    streams = gpd.read_file(streams_vect)
    im_slope = slope(read_image)
    x,y = np.where(im_slope < slope_threshold)
    exclusion = np.zeros((open_image.shape[0], open_image.shape[1]))
    for x, y in zip(x, y):
        exclusion[x, y] = 1
    buff = streams.buffer(stream_buff)
    buff = gpd.GeoDataFrame(geometry = buff, crs = streams.crs)
    buff['val'] = 2
    with open_image as src:
        feats = rasterio.features.rasterize(shapes = ((geom,value) for geom, value in zip(buff.geometry, buff['val'])), 
                                            out_shape = src.shape, 
                                            transform = src.transform, 
                                            fill = 0)
    x, y = np.where(feats + exclusion == 1)
    final_exclusion = np.zeros((open_image.shape[0], open_image.shape[1]))
    for x, y in zip(x, y):
        final_exclusion[x, y] = 1
    feature_radius = max_radius
    min_x, min_y, max_x, max_y = open_image.bounds
    x, y = np.where(final_exclusion == 1)
    #List of all points that are flat and not in a drainage
    point_list = [[x, y] for x, y in zip(x, y)]
    redo = True    
    while redo:    
        coord_list = random.sample(point_list, no_points)
        #Need to make sure none of these objects are touching each other
        test_gdf = gpd.GeoDataFrame(geometry = [Point(x) for x in coord_list], crs = open_image.crs)
        buffer_test = gpd.GeoDataFrame(geometry = test_gdf.buffer(feature_radius + 5), crs = open_image.crs)
        #Reference
        #https://gis.stackexchange.com/questions/457457/selecting-and-removing-polygons-that-overlap-within-a-single-geodataframe-in-pyt
        #Drop any features that overlap with the final exterior
        diss = buffer_test.dissolve()
        singleparts = gpd.GeoDataFrame(geometry = diss.apply(lambda x: [part for part in x.geometry.geoms], axis = 1).explode(), crs = diss.crs)
        singleparts["cluster"] = range(singleparts.shape[0])
        joined = gpd.sjoin(buffer_test, singleparts)
        joined = joined.merge(joined.groupby("cluster")["cluster"].count().rename("polycount"), left_on = 'cluster', right_index = True)
        joined = joined.loc[joined["polycount"] == 1]
        #Turn these buffered shapes back into points
        #Now the features should not overlap
        joined['centroid'] = joined['geometry'].centroid
        coord_list = [[x.x, x.y] for x in joined['centroid']]
        if len(coord_list) == no_points:
            redo = False
        else:
            redo = True
    shapely_points = []
    for i in coord_list:
        shapely_points.append(Point(i[1] + min_x, i[0] + min_y)) #This converts them to real points
    points = gpd.GeoDataFrame(geometry = shapely_points, crs = open_image.crs)
    feature_raster = rasterio.open(raster).read(1)
    below = []
    for i in range(0,len(points['geometry'])):
        below.append(points.iloc[i]['geometry'])
    #Use that to build the below_berm_poly list as in the random circles raster
    below_berm_poly = []
    below_berm_int_poly = []
    for i in below:
        feature_radius = random.uniform(min_radius, max_radius)
        berm = feature_radius + random.uniform(2, 4)
        berm_poly = shapely.buffer(i, berm)
        interior_poly = shapely.buffer(i, feature_radius)
        berm = noisy_poly(berm_poly)
        interior = noisy_poly(interior_poly)
        below_berm_poly.append(berm)
        below_berm_int_poly.append(interior)
    for i in below_berm_poly:
        with rasterio.open(raster) as src:
            berm = rasterio.features.rasterize(
            [i],
            out_shape = src.shape,
            transform = src.transform)
        x, y = np.where(berm == 1)
        x = x.tolist()
        y = y.tolist()
        #Berm height, based on target measurements
        height = random.uniform(0.2, 0.6)
        for x, y in zip(x, y):
            value = feature_raster[x, y] + height
            feature_raster[x, y] = value
    for i in below_berm_int_poly:
        with rasterio.open(raster) as src:
            berm = rasterio.features.rasterize(
            [i],
            out_shape = src.shape,
            transform = src.transform)
        x, y = np.where(berm == 1)
        x = x.tolist()
        y = y.tolist()
        height = random.uniform(0.1, 0.9)
        for x, y in zip(x, y):
            #Interior depth below grade, based on target measurements
            value = feature_raster[x, y] - height - 1
            feature_raster[x, y] = value
    exterior_gdf = gpd.GeoDataFrame(geometry = below_berm_poly, crs = open_image.crs)
    minx, miny, maxx, maxy = open_image.bounds
    coords = ((minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny))
    raster_bounds = gpd.GeoDataFrame(geometry = [Polygon(coords)], crs = open_image.crs)
    exterior_gdf = gpd.overlay(exterior_gdf, raster_bounds, how = 'intersection')
    with rasterio.open(raster) as src:
        annotation_mask = rasterio.features.rasterize(
            exterior_gdf['geometry'], 
            out_shape=src.shape,
            transform=src.transform)
    exterior_gdf['val'] = range(1, len(exterior_gdf.index) + 1)
    with rasterio.open(raster) as src:
        annotation_mask_RGB = rasterio.features.rasterize(
            shapes = ((geom,value) for geom, value in zip(exterior_gdf.geometry, exterior_gdf.val)), 
            out_shape=src.shape,
            transform=src.transform)
    return [annotation_mask, annotation_mask_RGB, feature_raster, exterior_gdf]