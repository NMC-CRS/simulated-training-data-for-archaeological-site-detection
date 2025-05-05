# Simulated training data for archaeological site detection
## Description
This repository holds the code we used to assess the efficacy of procedurally generated archaeological site images for training a deep learning model to detect archaeological sites in lidar data. The scripts included in the ```utils``` folder allow the user to create a procedurally generated dataset of tar-kiln-like objects in a lidar-derived DEM and train a Mask R-CNN model using a pretrained ResNet50 backbone.

## Installation
The scripts included here require Python 3.10 maximum. The workflows described below use the following packages: `pandas`, `rasterio`, `shapely`, `geopandas`, `matplotlib`, `numpy`, `scikit-learn`, `scikit-image`, `tensorboard`, `torch_snippets`, `albumentations`, `torchmetrics`, `pycocotools`, and `tifffile`. Scripts for all stages can be found in the ```utils``` folder.

## Procedural Generation

The script ```generate_unknown_objects.py``` contains the functions we used to generate the training datasets for our paper. ```create_tar_kiln_like_objects()``` corresponds to the paper's "Method 1" and ```create_simple_objects()``` corresponds to the paper's "Method 2." These scripts work by creating the vector shapes shown below using size ranges provided by the user and placing them in suitable areas as defined by an exclusion model: 

![Feature placement and vector creation overview](/docs/Figure_6.png)

The functions return a list of three `numpy` arrays (indices 0, 1, and 2) and a `geopandas` GeoDataFrame (index 3). The first array is a binary annotation mask of the simulated objects (0), the second array is an annotation mask of the simulated objects with each object represented by a unique integer (1), and the third array is the modified raster with procedurally generated features added. The final output (list index 3) is a `geopandas` GeoDataFrame with each added object as its own vector annotation. 

To use these DEMs with the MaskRCNN workflow described below, the second annotation mask (1) and the third array (2) should be exported as a georeferenced raster (at minimum; you may also wish to export the GeoDataFrame to a GeoPackage or shapefile to measure object by object performance in the test dataset). This can be accomplished [with rasterio](https://rasterio.readthedocs.io/en/stable/topics/writing.html) or [with GDAL](https://gdal.org/en/stable/tutorials/raster_api_tut.html) using the original input raster to set attributes like height, width, and data type.

## MaskRCNN Implementation

### File organization
These scripts use hard-coded relative paths. Data should be arranged in input and output folders in the following file structure:

```bash
├── CNN_data
│   ├── CNN_input
│   │   ├── Grids
│   │   ├── Input_[vis name]_[tile size]
│   │   ├── [Object name mask subfolder]
│   │   │   ├── Target_[buffer size]_[tile_size]_RGB
│   ├── CNN_output
│   │   ├── Model_predictions
│   │   ├── Model_weights
│   ├── Visualizations_and_annotations
│   │   ├── Rasters
│   │   ├── Shapefiles
```

where:\
`[Object name mask subfolder]` can be anything that relates to the object to detect (e.g., *Tar_kiln_masks*),\
`[vis name]` is the name of the visualization type, **without spaces** (e.g., *SLRM20m*),\
`[tile size]` is the height/width of the tiles (e.g., *256*), and\
`[buffer_size]` is the buffer around the annotated object, **without spaces** (e.g., *10m*)

### Workflow
The scripts included in the **utils** folder cover the 3 main steps of the deep learning workflow, which are described in more detail below:
1. Create training tiles
2. Train a model
3. Apply a trained model to new data

#### Create training tiles
To create our training dataset, we first created different visualization maps from our modified, lidar-derived DEM (exported from the output of one of the procedural generation functions described above) using the [Relief Visualization Toolbox](https://rvt-py.readthedocs.io/en/latest/rvtfor_qgis.html) in QGIS. Then, we used the `create_overlapping_grid.py` script to create two overlapping grids using one of the visualization raster as a basemap to get the grids' extent. We created a 256x256 grid and saved it in the **Grids** folder in **CNN_input**. 

We then used the `tile_raster_from_grid.py` script to tile each of our visualization maps. The resulting tiles were separated into input folders with names that reflected the visualization. For example, 256x256 pixel tiles created from the SLRM map using 20m moving window were placed into a folder called **Input_SLRM20m_256** within the **CNN_input** folder, whereas the 512x512 tiles created from the slope map were saved into a folder called **Input_Slope_512**. 

Similarly, we tiled the annotation mask using the same grids. The resulting tiles were placed in target folders with names that reflected their size and the buffer size around the objects. For example, the 256x256 tiles from the map with 20m buffers around terraces were saved in a folder called **Target_20m_256** within the **Terrace_masks** subfolder of **CNN_input**.

#### Train a model
This step relies mainly on the `maskrcnn_main_script.py` script, which calls the other MaskRCNN scripts.

In general, `maskrcnn_main_script.py` calls `calc_mask_perc.py` to determine which mask tiles overlap with at least one object (i.e., which tile has some pixels with value > 0). This part allows setting a pre-processing threshold which is used to ignore any mask tile with less positive pixels than that threshold. This prevents training on tiles that have only very small fractions of the object to detect. This defines a list of tiles that will be used in the training/validation/testing of the model. Tiles without object are completely ignored in this step.

Then, the script calls the `separate_datasets.py` script to separate the tiles into training/validation/testing datasets. The separation can be done randomly (80% training/10% validation/10% testing) or based on geographical bounds provided by the user. The script then calls the `clean_overlapping_tiles` function in the `clean_datasets.py` script to reassign overlapping tiles from different datasets. This results in 3 completely different datasets without overlap.

After separating and cleaning datasets, the script defines the data formatting and data loading workflow, which uses functions found in the `maskrcnn_dset_rgb.py` and `maskrcnn_transformations_rgb.py` scripts. 

The script then imports the the ResNet50 pre-trained backbone by calling the script in the **mask_rcnn_backbones** folder (this is currently the only backbone available in our implementation) and then calls the `set_lr_parameters.py` script to set the learning rate of the model parameters that will learn to a starting rate of 0.001 and set learning rate of the frozen parameters to 0.

When all of these pre-processing steps are done, the script creates a ***filename*** variable and prints it to the Console. This variable holds the name that will be used to save weights if necessary, as well as to create prediction maps in the 3rd step of the workflow. This name holds a lot of information in a specific order, which can be automatically parsed:

`MaskRCNN_{backbone}_{n_epochs}ep_{buffer_size}m_{loss_fun}_{batch_size}bs_{lr_type}_{vis1}_{vis2}_{vis3}_{threshold}Thresh_{im_size}_{time_stamp}`

* **model_structure**: "MaskRCNN"
* **backbone**: "ResNet50" (The pre-trained backbone structure)
* **n_epochs**: The number of times the model sees all the training tiles
* **buffer_size**: The size of the buffer around the annotated object
* **loss_fun**: "MaskRCNN_loss" (The loss function used to improve the model, which is a mix of multiple values)
* **batch_size**: The number of training tiles fed to the model at the same time
* **lr_type**: "lrVariable" (Because the learning rate decreases when validation loss stagnates)
* **vis1**: The name of the first visualization used
* **vis2**: The name of the second visualization used
* **vis3**: The name of the third visualization used
* **threshold**: Post-processing threshold (for 3rd step) that will automatically remove predicted objects smaller than this given value
* **im_size**: Size (height and width) of the training tiles
* **time_stamp**: Unique time stamp that ensures trainings run with the same parameters will not overwrite previous runs

Therefore, the filename ***MaskRCNN_ResNet50_20ep_5m_MaskRCNN_loss_8bs_lrVariable_SLRM20m_SLRM10m_Slope_100Thresh_256_1709159091*** says that this training was done using **ResNet50** pre-trained weights. The learning rate of the model changed when validation loss started stagnating. The model was trained on **256x256 pixel** images that combined **SLRM 20m**, **SLRM 10m**, and **Slope** visualizations fed in batches of **8 images**. The mask tiles used to teach the model were from **5m** buffers around the annotated objects. Finally, this training was done over **20 epochs**.

At that point, the script calls its `train_and_validate_model` functions, which loads the training tiles. This script pre-processes the tiles by combining the 3 visualizations into a 3-band tile, applying data augmentations, and computing bounding boxes around each mask. It compares its predictions to the actual masks and boxes to compute the loss, and goes back through its parameters to update their weights in order to diminish that loss. It then loads the validation tiles and pre-processes them (no augmentation for validation tiles, however) and runs them through the model to compute validation metrics. The `train_and_validate_model` function does this loop for the number of epochs specified by the user (```n_epochs```).

Finally, the script calls its `test_model` function, which runs the testing dataset through the trained model and computes metrics from it. If the user has decided to save the weights of that model, this is done after that step. The weights file is saved in the **CNN_output/Model_weights** folder.

##### Metrics calculated
In these scripts, we calculate the same metrics for training, validation, and testing datasets:
* **Recall**: The ratio of true positive pixels that are correctly predicted by the model
* **Precision**: The ratio of predicted pixels that are true positives
* **F1 score**: The harmonic mean of recall and precision

![Visual depiction of precision and recall](/docs/Precision_and_recall.png)

#### Apply a trained model to new data
This step relies mainly on the `apply_pretrained_model_to_new_data.py` script. This script has two `main` functions: `main_without_metrics` can be used to apply to new data where we do not have any annotated objects to compare the predictions to (so, new data completely), whereas `main_with_metrics` can be used to apply the trained model on the full map used to train it, for which we may have object annotations already. The latter uses functions from `calculate_metrics.py` to calculate metrics between the model predictions and a provided shapefile of annotations.

In general, the `apply_pretrained_model_to_new_data.py` script's `main` functions load the pre-trained weights to the appropriate model structure, and then run the new tiles through that model to produce predictions for each of those tiles. The tiles containing predictions are saved as GeoTIFFs in a folder within the **Model_predictions** subfolder of **CNN_output**. That folder is named after the ***filename***. The script then merges all those predicted tiles into one big raster, which is saved (with the same ***filename***) in **Model_predictions**. Finally, the script vectorizes that raster so that all adjacent positive pixels are grouped into polygons and saves it as a GeoPackage (with the same ***filename***). If an annotation GeoPackage and a post-processing threshold value are given, the script then deletes polygons with an area smaller than the provided threshold and then uses the overlap between the predicted polygons and the annotated polygons to compute **recall**, **precision**, and **F1 score** of objects rather than pixels.

***IMPORTANT NOTE***: To apply a pretrained model to new data, you need to make sure that the new data is in the same format as the trained data (visualizations and tile sizes). The new data should also be stored in the same file structure as shown above, as the model uses hard-coded relative paths and variable names to find those folders.

## Post-processing
The function ```assess_profile()``` in ```post_process.py``` iterates through a file of vector shapes and assesses the likelihood that each shape is a concave up or concave down object based on its major/minor axis profiles, as shown in the figure below:

![Concave-up and concave-down profiles](/docs/Figure_9.png)

In the paper, we use this function to exclude features that are likely mima mounds (a natural feature prevalent in the low-relief landscape of our study area). We also implemented a size exclusion and excluded all features that intersected or were within a certain distance of a drainage using QGIS, so no code is provided for those stages.

## Support
For support, contact one of the authors of this repository or open an issue.

<!-- Removing for now, will add back after peer review -->

<!-- ## Authors and acknowledgment
**Authors:**\
Claudine Gravel-Miguel, PhD\
Grant Snitker, PhD\
Katherine Peck, MS -->
