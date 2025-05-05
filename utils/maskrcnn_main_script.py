# -*- coding: utf-8 -*-

"""
Created on Thu Nov 16 13:08:45 2023

@author: Claudine Gravel-Miguel edited the code from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html to fit our needs.

@description: This script formats the training tiles, separates them to training/validation/testing, and trains and evaluate the model. 
A lot of the function it uses are here, but it also calls other scripts here and there to train a model and test it.

"""

# Set up of the necessary libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torch_snippets import Report # for Report

from torch.optim.lr_scheduler import StepLR # Addition to adjust lr

from torchmetrics.detection import MeanAveragePrecision

import pandas as pd

from pprint import pprint

import math

import sys

import time

import os

from datetime import datetime

import calendar

import numpy as np

import random

# Import functions from the web that I want to keep separate
import maskrcnn_eval_loss as mel

# Import the Mask RCNN utils file that I got from the PyTorch tutorial.
import mask_rcnn_utils as mutils

# Import my own functions
import calc_mask_perc as ct
import maskrcnn_dset_rgb as dset
import maskrcnn_transformations_rgb as A
import separate_datasets as sepdata
import set_lr_parameters as setlr
import clean_datasets as clean
import log_metrics as logmet

def mean_metrics(vals):
    '''
    This calculates the mean of all dictionaries.
    This is called by the evaluate function.

    Parameters
    ----------
    vals : list of dictionaries)
        Dictionaries created on the validation dataset.

    Returns
    -------
    results_dict : PyTorch dictionary
        Dictionary of the compiled IoU metrics.

    '''
    
    # Create an empty dictionary that will take on the compiled metrics
    results_dict = {}
    
    # Number of items in the vals list
    n_obs = len(vals)
    
    # Add the values in corresponding keys of all dictionaries and divide the sum by the number of dictionaries to get their mean average
    for val in vals:
        for k in val.keys():
            results_dict[k] = (results_dict.get(k, 0) + val[k]).item() # item.() transforms the tensor into a simple number
    
    # Add the calculated mean to the results dictionary
    for key in results_dict.keys():
        results_dict[key] = round(results_dict[key] / n_obs, 3)

    # Return the dictionary
    return results_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, log):
    '''
    This runs the training dataset (image and targets) through the model in batches, and computes different types of losses based on the predicted bounding boxes (set up by metric.logger())

    Parameters
    ----------
    model : PyTorch model
        The model structure being trained.
    optimizer : torch.optim optimizer
        Identifies the nodes that are learning and at which rate.
    data_loader : Pytorch DataLoader
        The dataloader for training dataset.
    device : str
        Name of the device on which the model is loaded (either "cuda" or "cpu").
    epoch : int
        Number of the epoch that runs this training.
    print_freq : int
        The frequency at which the metrics will be printed.
    log : torch_snippet object
        Object that logs and prints the values as the epoch trains. Allows following progress..

    Returns
    -------
    metric_logger : SmoothedValue object
        The object that holds all the values from that batch.

    '''
    
    # Number of training batches used to calculate updated metrics
    N_batch = len(data_loader)
    
    # Set the model on training mode
    model.train()
    print("\n***TRAINING MODE***\n")
    
    # Set up the metric logs that will be printed during the epoch
    metric_logger = mutils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", mutils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"\nEPOCH {epoch + 1}:"
    
    # Set up the batch counter
    batch = 0
    
    # Iterates through each batch
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):

        # Update the batch counter
        batch += 1
        
        # List the images in the batch and send them to the appropriate device (cuda or cpu)
        images = list(image.to(device) for image in images)
        
        # Get a dictionary of all targets for each image and send them to the appropriate device
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        # Reset the optimizer
        optimizer.zero_grad()
        
        # Run the batches of image and their target through the model and compile the losses calculated
        with torch.amp.autocast(device_type = "cuda" if torch.cuda.is_available() else "cpu", enabled = True):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Stop the model if there are erratic losses (sometimes happen when learning rate is too big)
        if losses > 100:
            print(f'\nProblem with the loss, it is too big: {losses}')
            sys.exit("Error message")
                    
        # Go backward through the model and update the nodes' weights and learning rates
        losses.backward()
        optimizer.step()
        
        # Reduce the values in the dictionary from all processes so that all processes have the average value
        loss_dict_reduced = mutils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        # Extract the numerical loss
        loss_value = losses_reduced.item()

        # Stop the model if loss is infinite (something is wrong)
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        # Update the metric logger with the latest losses and latest learning rate
        metric_logger.update(loss = losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])

        # For outputs and writer, calculate the average loss
        loss_avg = metric_logger.__getattr__("loss").global_avg
         
        # Output it
        log.record(epoch+(batch)/N_batch, trn_loss=loss_avg, end='\r')
    
    # Return the metric_logger
    return metric_logger

def validate(model, data_loader, device, epoch, log, message):
    '''
    This runs the validation/testing dataset and its associated target dataset through the trained model and computes the metrics by comparing its predictions to the actuals. 
    This is called at every epoch by train_and_validate_model to follow the training progress and avoid overfitting.
    It is also called once at the end of all epochs for ultimate testing. In that case, the dataset is the testing dataset.

    Parameters
    ----------
    model : PyTorch model
        The model structure being evaluated.
    data_loader : Pytorch DataLoader
        The dataloader for validation/testing dataset.
    device : str
        Name of the device on which the model is loaded (either "cuda" or "cpu").
    epoch : int
        Number of the epoch that runs this validation.
    log : torch_snippet object
        Object that logs and prints the values as the epoch trains. Allows following progress.
    message : str
        String to keep track of progress. When called for validation dataset, it prints "***VALIDATION MODE***", whereas it prints "***TESTING MODE***" when called for testing.

    Returns
    -------
    val_mAP : double
        Average precision computed on validation.
    val_mAR : double
        Average recall computed on validation.
    val_f1 : double
        Average F1 computed on validation.
    bbox_res : list of dictionaries
        Dictionaries created on the validation dataset.

    '''
   
    # Print the validation or testing message to Console.
    print(message)
    
    # Number of training batches
    N_batch = len(data_loader)
    
    # Define the CPU device, used to compute the metrics
    cpu_device = torch.device("cpu")
    
    # Set up the metric logs that will be printed during the epoch
    metric_logger = mutils.MetricLogger(delimiter="  ")
    header = f"\nEPOCH {epoch + 1}:"

    # Set up the tool that will compute the metrics of the bounding boxes
    metric_bbox = MeanAveragePrecision(iou_type="bbox", 
                                       iou_thresholds = [0.5],
                                       rec_thresholds = [0.5])
    
    # Set up an empty list that will record the validation bounding boxes' metrics (mAP and mAR) at each batch.
    vals_bbox = []
    
    # Set the batch counter for printing purposes
    batch = 0
    
    # Set up variables to get average metrics of all batches in each epoch
    total_val_recall = 0
    total_val_precision = 0
    
    # Set the model on evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Iterate through each batch
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            
            # Update the batch counter
            batch += 1
            
            # List the images in the batch and send them to the appropriate device
            # We do not need to send the targets to cuda, because they will be used on cpu later on.
            images = list(img.to(device) for img in images)
            
            # Wait for all kernels in all streams on a CUDA device to complete.
            if torch.cuda.is_available():
                torch.cuda.synchronize() 
            
            # Set up the time when the model started running through the images (to know how long it takes per batch)
            model_time = time.time()
            
            # Run the validation images through the model
            outputs = model(images)
            
            # Send the resulting predictions to CPU
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            
            ## Compute metrics manually
            
            ## BBOX metrics
            metric_bbox.update(outputs, targets)
            metric_bbox_results = metric_bbox.compute()
            metric_bbox.reset() # to avoid memory problems
            
            # Extract precision and recall 
            map_50 = metric_bbox_results['map'].detach().numpy() # mean precision at IoU threshold 0.5
            total_val_precision += map_50
            
            mar_100 = metric_bbox_results['mar_100'].detach().numpy() # mean recall given 100 detections per image
            total_val_recall += mar_100
            
            # Compute f1 score from precision and recall (the 0.0001 is to avoid divisions by 0)
            total_val_f1 = (2 * total_val_recall * total_val_precision + 0.0001)/(total_val_recall + total_val_precision + 0.0001)  
            
            # Update the averages per batch for logging
            val_mAP = total_val_precision / batch
            val_mAR = total_val_recall / batch
            val_f1 = total_val_f1 / batch
            
            # log the 0.5 mAP and medium mAR to print to Console
            log.record(epoch + batch / N_batch, val_mAP = val_mAP, val_mAR = val_mAR, val_f1 = val_f1, end='\r')
            
            # Append the metrics to the list
            vals_bbox.append(metric_bbox_results)
            
            # Update the time the model worked on predictions and computed metrics and print it
            model_time = time.time() - model_time
            metric_logger.update(model_time = model_time)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # Call the above function to compute one average for all batches
    bbox_res = mean_metrics(vals_bbox)

    # Return the metrics that we are interested in and want logged and/or printed
    return val_mAP, val_mAR, val_f1, bbox_res

def train_and_validate_model(device, model, n_epochs, trn_dl, val_dl, optimizer, writer, save_weights, output_path, filename):
    '''
    Iterates through epochs to feed training images to the model (in batches), predict objects on those images, and calculates bounding box loss (compared with actual boxes)
    Updates the weights of the model to mininize the loss.
    Then runs all validation images (and their masks and bounding boxes) through the model to calculate loss on validation dataset. This does not train the model.
    Finally, reruns the validation images through the model to calculate recall, precision, and F1 score.

    Parameters
    ----------
    device : str
        The device where the training will occur ("cuda" or "cpu").
    model : Pytorch model
        Pytorch model structure.
    n_epochs : int
        Number of epochs to train the model. The model sees all the training images in each epoch.
    trn_dl : Pytorch Dataloader
        The workflow through which the training images and their targets go through when passed to the model.
    val_dl : Pytorch Dataloader
        The workflow through which the validation images and their targets go through when passed to the model.
    optimizer : Pytorch optimizer
        Optimizer of the learning parameters, which gets updated during training.
    writer : bool
        If True, the code creates a Tensorboard file and log metrics to it.
    save_weights : bool
        If True, the code saves the weights to a .pt file.
    output_path : str
        Path where to save the weights.
    filename : str
        The name that will be assigned to the saved weights.

    Returns
    -------
    None.

    '''
    
    # This was the original tutorial's scheduler and it may work for some circumstances, so keeping the code here.
    scheduler = StepLR(optimizer,
                       step_size=3,
                       gamma=0.1)
    
    # To log how long training took
    start_time = datetime.now()
    
    # Set up the log to follow progress
    log = Report(n_epochs)
    
    # Set variables that will be used to verify if the model is improving (if it is, it might save the weights)
    # Set unrealistic values at first that will be overwritten by the first epoch with better values
    tr_loss_prev = 9999
    val_mAR_prev = 0
    
    # Iterate through for each epoch
    for epoch in range(n_epochs):
        
        # For cleaner code below
        current_epoch = epoch + 1
        
        # Train for one epoch
        metric_logger = train_one_epoch(model, optimizer, trn_dl, device, epoch, print_freq = len(trn_dl) + 1, log = log)
        
        # Calculate the average loss for outputs and writer
        tr_loss_avg = metric_logger.__getattr__("loss").global_avg
        
        # Run validation images through the model just to compute validation loss (to update lr). This does not train the model.
        # no_grad() reduces the burden on memory.
        with torch.no_grad():
            print("\n***COMPUTE VAL LOSS***")
            val_loss = mel.evaluate_loss(model, val_dl, device)
        
        # Evaluate on the validation dataset
        val_mAP, val_mAR, val_f1, bbox_res = validate(model, val_dl, device, epoch, log, message = "\n***VALIDATION MODE***\n")
        
        # For outputs (no need to update at every batch)
        if writer is not None:
            writer.add_scalar("Loss/train", tr_loss_avg, current_epoch)
            writer.add_scalar("Loss/val", val_loss, current_epoch)
            writer.add_scalar("Precision/val", val_mAP, current_epoch)
            writer.add_scalar("Recall/val", val_mAR, current_epoch)
            writer.add_scalar("F1/val", val_f1, current_epoch)
        
        # Change the learning rate if scheduler says so
        scheduler.step()
        
        # to print the lr and the time the training has been taking
        curr_lr = scheduler.get_last_lr()
        time_since_start = datetime.now() - start_time
        
        # Some more detailed metrics are printed at the first and last epochs, as well as every 10 epochs in between.
        if (current_epoch) == 1 or (current_epoch)%10 == 0 or current_epoch == n_epochs:
            
            # These are classic bounding box MAP metrics
            print("\nIoU metric: bbox")
            pprint(bbox_res)

            print(f'\nLR: {curr_lr}, time since start: {time_since_start}')
            print(f'TRAIN loss: {tr_loss_avg:.3f}')
            print(f'VAL loss: {val_loss:.3f}, recall: {val_mAR:.3f}, precision: {val_mAP:.3f}, F1: {val_f1:.3f}')
        
        # At the end of the training, cleaned metrics are printed
        if current_epoch == n_epochs:
            logmet.populate_metrics(output_path, filename, "Duration_min", round(time_since_start.seconds/60, 0))
            
            logmet.populate_metrics(output_path, filename, "Train_loss", round(tr_loss_avg, 3))
            
            logmet.populate_metrics(output_path, filename, "Val_loss", round(val_loss.item(), 3))
            logmet.populate_metrics(output_path, filename, "Val_recall", round(val_mAR, 3))
            logmet.populate_metrics(output_path, filename, "Val_precision", round(val_mAP, 3))
            logmet.populate_metrics(output_path, filename, "Val_F1", round(val_f1, 3))
        
        # Check if the training loss and recall are smaller than in previous epochs.
        # If they are, the code will save the weights if save_weights is True.
        if tr_loss_avg < tr_loss_prev and val_mAR >= val_mAR_prev:
            
            # update the metrics for next epoch.
            tr_loss_prev = tr_loss_avg
            val_mAR_prev = val_mAR
            
            # Save (or overwrite) the weights to a .pt file.
            if save_weights:
                print(f'\n*IMPORTANT*: weights saved at epoch {current_epoch}')
                torch.save(model.state_dict(), f"{output_path}/Model_weights/{filename}.pt")
    
    # Force summary writer to send any buffered data to storage
    if writer is not None:
        writer.flush()
        
    return

def test_model(device, model, train_dir_dim1, train_dir_dim2, train_dir_dim3, mask_dir, inputs_test, batch_size, filename, output_path):
    '''
    Runs the testing dataset through the trained model and compute loss, recall, precision, and F1 values from its predictions.

    Parameters
    ----------
    device : str
        The device where the training will occur ("cuda" or "cpu").
    model : TYPE
        Pytorch model.
    train_dir_dim1 : str
        Path to the tiles of the first visualization..
    train_dir_dim2 : str
        Path to the tiles of the second visualization..
    train_dir_dim3 : str
        Path to the tiles of the third visualization..
    mask_dir : str
        Path to the mask tiles.
    inputs_test : TYPE
        List of the filenames of the testing dataset.
    batch_size : TYPE
        Number of images that will be uploaded to the model at the same time.
    filename : str
        Name of the weight file
    output_path : str
        Path of the CNN_output folder in which to save the geopackages

    Returns
    -------
    None.

    '''
    
    # Assign each set of testing image to their appropriate folder
    inputs_test_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_test] 
    inputs_test_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_test] 
    inputs_test_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_test] 
    targets_test = [f'{mask_dir}/{item}' for item in inputs_test] 
    
    # Format the testing dataset into tensors of the proper format
    test_ds = dset.MaskBoxDataset(inputs_test_dim1, inputs_test_dim2, inputs_test_dim3, targets_test)
    
    # Workflow to load the images in batches
    test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle = True, collate_fn = A.collate_fn)
    
    # Reset the log 
    log = Report(0)
    
    # Run the testing dataset through the evaluate function
    tst_mAP, tst_mAR, tst_f1, bbox_res = validate(model, test_dl, device, 0, log, message = "\n***TESTING MODE***\n")
        
    # Log and print metrics.
    print(f'TEST recall: {tst_mAR:.3f}, precision: {tst_mAP:.3f}, F1: {tst_f1:.3f}')
    
    logmet.populate_metrics(output_path, filename, "Test_recall", round(tst_mAR, 3))
    logmet.populate_metrics(output_path, filename, "Test_precision", round(tst_mAP, 3))
    logmet.populate_metrics(output_path, filename, "Test_F1", round(tst_f1, 3))
    
    return

def main(backbone, vis1, vis2, vis3, im_size, buffer_size, data_path, mask_folder_name, threshold, batch_size, separation_random, train_bounds, n_epochs, log_metrics, save_weights, output_path, lr_variable, remove_overlap):
    '''
    Calls all the functions to load the model structure, format the different datasets, train the model, and then evaluate it.
    This function is called by the 02_one_script_to_rule_them_all.py script.

    Parameters
    ----------
    backbone : str
        UNUSED. Placeholder so that this can be called in 02_one_script.
    vis1 : str
        Name of the first visualization. This is to find the correct folder to upload.
    vis2 : str
        Name of the second visualization. This is to find the correct folder to upload.
    vis3 : str
        Name of the third visualization. This is to find the correct folder to upload.
    im_size : int
        Size of the training tiles' height (or width).
    buffer_size : int
        Buffer size (in meters) around the annotated object. This is to find the correct folder to upload.
    data_path : str
        Path to the CNN_input folder that holds inputs folders as well as the mask subfolder identified in mask_folder_name (see below).
    mask_folder_name : str
        Name of the folder that holds the masks folders. This is to find the correct folder to upload.
    threshold : int
        For pre-processing. This removes any mask images that have less than the threshold's number of positive's values.
    batch_size : int
        Number of images that will be uploaded to the model at the same time.
    separation_random : bool
        If set to True, the training/validation/testing datasets are separated randomly (80-10-10). If False, they are separated geographically based on train_bounds (see below).
    train_bounds : list
        xmin, ymin, xmax, and ymax around the tiles that will be used for training.
    n_epochs : int
        Number of epochs to train the model. The model sees all the training images in each epoch.
    log_metrics : bool
        If True, the code creates a Tensorboard file and log metrics to it.
    save_weights : bool
        If True, the model saves the trained weights to a .pt file.
    output_path : str
        Path to the CNN_output folder in which the predicted tiles, compiled tiff, compiled shapefiles will be saved.
    lr_variable : bool
        If the learning rate is updating when val_loss stagnates (True) or kept constant at 0.001 (False).
    remove_overlap : bool
        If we clean the datasets to remove any overlap between them

    Returns
    -------
    filename : str
        Name that is used for weights file, the folder that holds predicted tiles, the compiled geotiff, and the compiled shapefile.
    inputs_test : list
        List of the testing dataset to use to compute object-by-object metrics when weights are saved (done by another script).

    '''
    
    # Define the device from the start
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility and comparability
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # Define paths to each visualization dataset
    train_dir_dim1 = os.path.join(data_path, f'Input_{vis1}_{im_size}')
    train_dir_dim2 = os.path.join(data_path, f'Input_{vis2}_{im_size}')
    train_dir_dim3 = os.path.join(data_path, f'Input_{vis3}_{im_size}')
    mask_dir = os.path.join(data_path, f'{mask_folder_name}/Target_{buffer_size}m_{im_size}_RGB') 
    
    # REFINE THE DATASET OF TILES

    # Create a table with the size of annotations in each mask tile
    tiles_table = ct.log_mask_tiles(mask_dir)
    
    # Get the list of tiles with big enough annotations from the table created above
    filtered_df = tiles_table[tiles_table['min_nonzero'] > threshold]
    filenames = filtered_df['filename'].values.tolist()

    # Assign tiles to training or validation/testing datasets
    inputs_train, inputs_val, inputs_test = sepdata.separate_dataset(filenames, separation_random, train_bounds)

    if remove_overlap:
        ## Clean the datasets
        # Calculate the min_distance based on the resolution and im_size
        min_distance = clean.calculate_min_distance(f'{train_dir_dim1}/{inputs_train[0]}')
        
        # Use that min_distance to clean overlapping tiles
        inputs_train, inputs_val = clean.clean_overlapping_tiles(inputs_train, inputs_val, min_distance)
        inputs_train, inputs_test = clean.clean_overlapping_tiles(inputs_train, inputs_test, min_distance)
        inputs_val, inputs_test = clean.clean_overlapping_tiles(inputs_val, inputs_test, min_distance)

    # Calculate the number of cleaned tiles used.
    n_tiles_used = len(inputs_train) + len(inputs_val) + len(inputs_test)

    # Print info on the number of tiles used in each dataset
    print(f"\nUsing {n_tiles_used} tiles with objects in them.")
    print(f"Out of the {n_tiles_used} tiles, {len(inputs_train)}({round(len(inputs_train)/n_tiles_used, 2)}) are for training, {len(inputs_val)}({round(len(inputs_val)/n_tiles_used,2)}) are for validation, and {len(inputs_test)}({round(len(inputs_test)/n_tiles_used,2)}) for testing.")

    # Add the paths to the filenames in each category
    # TRAINING
    inputs_train_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_train] 
    inputs_train_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_train] 
    inputs_train_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_train] 
    targets_train = [f'{mask_dir}/{item}' for item in inputs_train] 

    # VALIDATION
    inputs_val_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_val] 
    inputs_val_dim2 = [f'{train_dir_dim2}/{item}' for item in inputs_val] 
    inputs_val_dim3 = [f'{train_dir_dim3}/{item}' for item in inputs_val] 
    targets_val = [f'{mask_dir}/{item}' for item in inputs_val] 
    
    # CREATE THE DATALOADER

    # Workflow to create the PyTorch dataset (these are still numpy arrays)
    trn_ds = dset.MaskBoxDataset(inputs_train_dim1, inputs_train_dim2, inputs_train_dim3, targets_train, transform = A.train_augmentation(im_size))
    val_ds = dset.MaskBoxDataset(inputs_val_dim1, inputs_val_dim2, inputs_val_dim3, targets_val)

    # Workflow to load the images in batches
    trn_dl = DataLoader(trn_ds, batch_size = batch_size, shuffle = True, collate_fn = A.collate_fn, num_workers = 0)
    val_dl = DataLoader(val_ds, batch_size = batch_size, shuffle = True, collate_fn = A.collate_fn, num_workers = 0)

    # CREATE THE MODEL STRUCTURE

    # Import the model backbone and pretrained weights
    from mask_rcnn_backbones.mask_rcnn_resnet50 import build_model
    
    # Define the number of classes (should be 2 for background and one type of object)
    num_classes = 2
    
    # Upload the model from the mask_rcnn script
    model = build_model(num_classes)
    
    # move model to device
    model.to(device)
    
    # Set the LR parameters
    pretrained_optimizer, new_optimizer = setlr.setup_parameter_learning_rate(model)
    
    # Needs to be variable to keep loss stable 
    lr_variable = True
    lr_type = "lrVariable"

    # DEFINE THE FILENAME

    # Define the timestamp for the filename
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # Define the filename from parameter values
    filename = f"MaskRCNN_ResNet50_{n_epochs}ep_{buffer_size}m_MAP_{batch_size}bs_{lr_type}_{vis1}_{vis2}_{vis3}_{threshold}Thresh_{im_size}_{time_stamp}"
    print(f'The weight file name of this training is: {filename}')
    
    # Define the writer if necessary
    if log_metrics:
        writer = SummaryWriter(log_dir = f"{output_path}/Tensorboard_files/{filename}")
    else:
        writer = None
    
    # Populate the log metrics file (create it first doesn't already exists)
    if separation_random:
        train_bounds = "Random"
        
    logmet.populate_params(output_path, "MaskRCNN", backbone, lr_variable, "MaskRCNN_loss", batch_size, vis1, vis2, vis3, 
                           threshold, len(inputs_train), len(inputs_val), len(inputs_test), remove_overlap, 
                           buffer_size, mask_folder_name, train_bounds, n_epochs, time_stamp)
    
    '''
    TRAIN AND VALIDATE MODEL
    '''
    
    train_and_validate_model(device, 
                model, 
                n_epochs, 
                trn_dl, 
                val_dl, 
                new_optimizer, 
                writer,
                save_weights, 
                output_path, 
                filename)

    if log_metrics:
        writer.close()
    
    '''
    TEST MODEL
    '''
    
    # Evaluate on testing dataset
    test_model(device, model, train_dir_dim1, train_dir_dim2, train_dir_dim3, mask_dir, inputs_test, batch_size, filename, output_path)

    # Get the list of test tiles' paths for the polygons.
    inputs_test_dim1 = [f'{train_dir_dim1}/{item}' for item in inputs_test]
    # Create a polygon that represents each dataset and write to disk
    gdf_train_polygons = clean.create_poly_from_tiles(inputs_train_dim1, "Training")
    gdf_val_polygons = clean.create_poly_from_tiles(inputs_val_dim1, "Validation")
    gdf_test_polygons = clean.create_poly_from_tiles(inputs_test_dim1, "Testing")

    # Join the geodatabases of dataset areas and export to disk
    joined_gdf = pd.concat([gdf_train_polygons, gdf_val_polygons, gdf_test_polygons])
    joined_gdf.to_file(f"{output_path}/Model_predictions/{filename}_areas.gpkg", driver="GPKG")

    # Return the filename (holds lots of metadata) and the inputs_test list, which can be used to create a map of predictions on test data
    return filename, inputs_test

# THE END

# Usage example

main("ResNet50", #Backbone (currently only ResNet50 is supported)
     "Slope", "Slope", "Slope", #vis1, vis2, and vis3 labels
     256, #Image size (pixels)
     0, #buffer size
     '/CNN_input', #Enter folder path to CNN_input folder here
     "Tar_kilns", #Enter name of mask folder
     0, #Threshold for including object in training
     8, #Batch size
     True, #Separation random (True) or geographically (False)
     [0,0,1,1], #Bounding box for training (if geographic)
     5, #Number of epochs (iterations) for training
     True, #Create a Tensorboard file to log metrics?
     True, #Save weights (.pt file)?
     '/CNN_output', #Enter folder path to CNN_output folder here
     True, #Variable (True) or constant (False) learning rate
     False) #Remove overlap?