import geopandas as gpd
import shapely
from shapely.geometry import LineString
import rasterio
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#These functions are called by the main assess_profile() function below

def polynomial_regression(x:list, y:list, degree:int) -> list:
    """
    Calculate the polynomial regression of an input distribution

    Parameters

    ----------
    x : list
        distribution x values (in this case, distance from line origin)
    y : list
        distribution y values (in this case, elevation)
    degree : int
        polynomial degree for regression calculation

    Returns
    -------
    list
        index 0: x values
        index 1: transformed y values
    """
    #Use scikit-learn Polynomial Features implementation to return a smoothed profile
    poly = PolynomialFeatures(degree = degree, include_bias = False)
    x = np.array(x)
    poly_features = poly.fit_transform(x.reshape(-1, 1))
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y)
    y_predicted = poly_reg_model.predict(poly_features)
    return [x, y_predicted]

def calculate_axes(shape:shapely.Polygon) -> list:
    """
    Given a polygon, calculate the major/minor axes of the minimum bounding rectangle

    Parameters
    ----------
    shape : shapely.Polygon
        Polygon to determine axes

    Returns
    -------
    list
        List of LineStrings representing the major/minor axes
    """
    #Get minimum rotated rectangle points
    mbr_points = list(zip(*shape.minimum_rotated_rectangle.exterior.coords.xy))
    #To sample along the axis, need to construct these
    upperleft = [x for x in mbr_points[0]]
    lowerleft = [x for x in mbr_points[1]]
    lowerright = [x for x in mbr_points[2]]
    uperright = [x for x in mbr_points[3]]
    #Construct sides of rectangle
    left = LineString([upperleft, lowerleft])
    right = LineString([uperright, lowerright])
    top = LineString([uperright, upperleft])
    bottom = LineString([lowerright, lowerleft])
    #Use line centroids to construct axes across center of shape
    axis_1 = LineString([left.centroid, right.centroid])
    axis_2 = LineString([top.centroid, bottom.centroid])
    return [axis_1, axis_2]

def calculate_profile(a:shapely.LineString, raster, distance_delta:float) -> list:
    """
    Calculate the elevation profile along a given line

    Parameters
    ----------
    a : shapely.LineString
        Line along which to calculate the elevation
    raster : _type_
        Underlying raster to sample (Warning: if line extends off raster, values will still be provided but they will be erroneous)
    distance_delta : float
        How frequently should the raster be sampled along the line (smaller number = more detailed but potentially more noisy data)

    Returns
    -------
    list
        Index 0: distance value from 0 - x along the line
        Index 1: Values at each distance
    """
    open_raster = rasterio.open(raster)
    #Get equally spaced points along the length of the input LineString every distance_delta
    distances = np.arange(0, a.length, distance_delta)
    points = [a.interpolate(distance) for distance in distances] + [a.boundary.geoms[1]]
    multipoint = shapely.ops.unary_union(points)
    points = gpd.GeoDataFrame(crs = open_raster.crs, geometry = [multipoint])
    #Merging these points creates a multi-geometry - explode to get individual x and y values
    coord_list = [(x, y) for x, y in zip(points['geometry'].explode(index_parts = True).x, points['geometry'].explode(index_parts = True).y)]
    new_points = gpd.GeoDataFrame(crs = open_raster.crs, geometry = gpd.points_from_xy(points['geometry'].explode(index_parts = True).x, points['geometry'].explode(index_parts = True).y))
    new_points['value'] = [x for x in open_raster.sample(coord_list)]
    if len(new_points['value']) == len(distances):
        d = distances.tolist()
    else: #Add an extra point at the end if the np.arange() fell short of the final length of the feature
        distances = np.append(distances, [a.length])
        d = distances.tolist()
    new_points['dist'] = d
    distances = new_points['dist'].tolist()
    values = [x[0] for x in new_points['value'].tolist()]
    return [distances, values]

#Main function

def assess_profile(shape:str, raster:str) -> gpd.GeoDataFrame:
    """
    Given a vector file of shapes, assess the likelihood that each shape is a concave up or concave down object based on its major/minor axis profiles

    Parameters
    ----------
    shape : str
        Vector file (e.g., ESRI shapefile, GeoPackage) file path
    raster : str
        DEM file path that will be sampled to test the profiles

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of shapes with added "type" column defining if shape is concave down (n shaped), concave up (u shaped), or ambiguous (could not be determined)
    """
    gdf = gpd.read_file(shape)
    gdf['type'] = ""
    for index, row in gdf.iterrows():
        shape = row['geometry']
        a = calculate_axes(shape)
        deg = 2
        p1 = calculate_profile(a[0], raster, .1)
        p2 = calculate_profile(a[1], raster, .1)
        if len(p1) == 1 or len(p2) == 1:
            gdf.at[index, 'type'] = 'ambiguous'
            print(str(index) + ' had unmatched indices, and the shape is thus considered ambiguous')
            pass
        else:
            d = polynomial_regression(p1[0], p1[1], deg)
            x = d[0]
            y = d[1]
            dydx = np.gradient(y, x) #First derivative
            d2ydx2 = np.gradient(dydx, x) #Second derivative
            trutharray = d2ydx2 < 0 #Check if the 2nd derivative is negative
            test1 = np.all(trutharray)
            d1 = polynomial_regression(p2[0], p2[1], deg)
            x = d1[0]
            y = d1[1]
            dydx = np.gradient(y, x) #First derivative
            d2ydx2 = np.gradient(dydx, x) #Second derivative
            trutharray = d2ydx2 < 0 #Check if the 2nd derivative is negative
            test2 = np.all(trutharray)
            if ((test1 == True) and (test2 == True)): #If both lines have negative second derivates, the shape is concave down (n shaped)
                gdf.at[index, 'type'] = 'concave down'
            elif ((test1 == False) and (test2 == False)):
                gdf.at[index, 'type'] = 'concave up' #If both lines have positive, the shape is concave up (u shaped)
            elif ((test1 == False) and (test2 == True)):
                gdf.at[index, 'type'] = 'ambiguous' #If the lines have different values, the type cannot be determined
            elif ((test1 == True) and (test2 == False)):
                gdf.at[index, 'type'] = 'ambiguous' #If the lines have different values, the type cannot be determined
    return gdf