# %%                        Packages
import os
import sys
import cv2
import csv
# import g4f
import json
import glob
import time
import ctypes
# import PyPDF2
import pickle
import plotly
import shutil
import zipfile
import imageio
import skimage
import logging
import requests
import warnings
import tempfile
# import anthropic
import threading
import webbrowser
import subprocess
# import langdetect
import numpy as np
import pandas as pd
import pydicom as dc
import nibabel as nib
# import tkinter as tk
from PIL import Image
from fpdf import FPDF
# import tensorflow as tf
from scipy import stats
import plotly.io as pio
from pathlib import Path
from scipy import ndimage
# from tkinter import font
# import ttkbootstrap as ttk
import plotly.express as px
from datetime import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.graph_objects as pxf
from skimage.transform import rescale
from numpy.f2py.auxfuncs import errmess
# from scipy.spatial.distance import cdist
# from tkinter import filedialog, messagebox
from skimage.color import gray2rgb, rgb2gray
from sklearn.neighbors import NearestNeighbors
# from tensorflow.keras.models import load_model
# from flask import Flask, request, jsonify, send_file
from scipy.ndimage import gaussian_filter, binary_dilation, label, distance_transform_edt
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects


# %% [markdown]
# ### Functions

# %%

def enhanced_adjacency_finder_with_ventricles(abnormal_wmh, normal_wmh, ventricle_mask, 
                                            min_area=5, adjacency_threshold=1, alignment_threshold=2,
                                            voxel_size=(1.0, 1.0)):
    """
    Enhanced adjacency finder that considers ventricle context for WMH reclassification.
    
    Parameters:
    -----------
    abnormal_wmh : numpy.ndarray
        Binary mask of abnormal WMH objects
    normal_wmh : numpy.ndarray  
        Binary mask of normal WMH objects (juxtaventricular)
    ventricle_mask : numpy.ndarray
        Binary mask of ventricle regions
    min_area : int
        Minimum area threshold for objects (in pixels)
    adjacency_threshold : float
        Distance threshold for adjacency (in mm, will be converted to pixels)
    voxel_size : tuple
        Voxel spacing in mm (height, width)
    
    Returns:
    --------
    pixels_to_move : numpy.ndarray
        Binary mask of pixels to move from normal_wmh to abnormal_wmh
    """
    
    # Convert adjacency threshold from mm to pixels
    pixel_threshold = adjacency_threshold / min(voxel_size)
    
    # Step 1: Remove small objects from abnormal_wmh
    labeled_abnormal, num_abnormal = label(abnormal_wmh)
    processed_abnormal = np.zeros_like(abnormal_wmh, dtype=bool)
    
    for i in range(1, num_abnormal + 1):
        object_area = np.sum(labeled_abnormal == i)
        if object_area >= min_area:
            processed_abnormal[labeled_abnormal == i] = True
    
    # Step 2: Find normal WMH objects adjacent to abnormal WMH
    dilated_abnormal = binary_dilation(processed_abnormal, 
                                     structure=np.ones((int(2*adjacency_threshold+1), int(2*adjacency_threshold+1))))
    
    # Get normal WMH objects that are adjacent to abnormal WMH
    adjacent_seed = dilated_abnormal & normal_wmh
    labeled_normal, num_normal = label(normal_wmh)
    
    adjacent_normal_objects = np.zeros_like(normal_wmh, dtype=bool)
    for j in range(1, num_normal + 1):
        if np.any(adjacent_seed[labeled_normal == j]):
            adjacent_normal_objects[labeled_normal == j] = True
    
    # Step 3: For each adjacent normal object, determine which pixels are 
    # positioned between abnormal WMH and ventricles
    pixels_to_move = np.zeros_like(normal_wmh, dtype=bool)
    
    # Re-label the adjacent normal objects for individual processing
    labeled_adjacent, num_adjacent = label(adjacent_normal_objects)
    
    for obj_id in range(1, num_adjacent + 1):
        current_object = (labeled_adjacent == obj_id)
        
        # Find pixels in this object that are between abnormal WMH and ventricles
        between_pixels = find_pixels_between_regions(
            current_object, processed_abnormal, ventricle_mask, voxel_size, alignment_threshold
        )
        
        pixels_to_move |= between_pixels
    
    return pixels_to_move

def find_pixels_between_regions(normal_object, abnormal_wmh, ventricle_mask, voxel_size, alignment_tolerance):
    """
    Find pixels in normal_object that are spatially between abnormal_wmh and ventricles.
    
    Parameters:
    -----------
    normal_object : numpy.ndarray
        Binary mask of a single normal WMH object
    abnormal_wmh : numpy.ndarray
        Binary mask of abnormal WMH regions
    ventricle_mask : numpy.ndarray
        Binary mask of ventricle regions
    voxel_size : tuple
        Voxel spacing in mm
        
    Returns:
    --------
    between_pixels : numpy.ndarray
        Binary mask of pixels that are between abnormal WMH and ventricles
    """
    
    between_pixels = np.zeros_like(normal_object, dtype=bool)
    
    # Get coordinates of pixels in the normal object
    normal_coords = np.where(normal_object)
    if len(normal_coords[0]) == 0:
        return between_pixels
    
    normal_points = np.column_stack(normal_coords)
    
    # Get coordinates of abnormal WMH and ventricle boundaries
    abnormal_coords = np.where(abnormal_wmh)
    ventricle_coords = np.where(ventricle_mask)
    
    if len(abnormal_coords[0]) == 0 or len(ventricle_coords[0]) == 0:
        return between_pixels
    
    abnormal_points = np.column_stack(abnormal_coords)
    ventricle_points = np.column_stack(ventricle_coords)
    
    # Scale coordinates by voxel size for accurate distance calculation
    normal_scaled = normal_points * np.array(voxel_size)
    abnormal_scaled = abnormal_points * np.array(voxel_size)
    ventricle_scaled = ventricle_points * np.array(voxel_size)
    
    # For each pixel in normal object, check if it's between abnormal WMH and ventricle
    for i, normal_pixel in enumerate(normal_scaled):
        # Find the closest points in abnormal WMH and ventricle
        dist_to_abnormal = np.min(np.linalg.norm(abnormal_scaled - normal_pixel, axis=1))
        dist_to_ventricle = np.min(np.linalg.norm(ventricle_scaled - normal_pixel, axis=1))
        
        # Find the closest abnormal and ventricle points
        closest_abnormal_idx = np.argmin(np.linalg.norm(abnormal_scaled - normal_pixel, axis=1))
        closest_ventricle_idx = np.argmin(np.linalg.norm(ventricle_scaled - normal_pixel, axis=1))
        
        closest_abnormal = abnormal_scaled[closest_abnormal_idx]
        closest_ventricle = ventricle_scaled[closest_ventricle_idx]
        
        # Check if the normal pixel lies approximately on the line between 
        # the closest abnormal and ventricle points
        if is_point_between(normal_pixel, closest_abnormal, closest_ventricle, tolerance=alignment_tolerance):
            between_pixels[normal_coords[0][i], normal_coords[1][i]] = True
    
    return between_pixels

def is_point_between(point, point1, point2, tolerance=2.0):
    """
    Check if a point lies approximately between two other points.
    
    Parameters:
    -----------
    point : numpy.ndarray
        The point to check
    point1, point2 : numpy.ndarray
        The two reference points
    tolerance : float
        Distance tolerance in mm
        
    Returns:
    --------
    bool : True if point is between point1 and point2
    """
    
    # Calculate distances
    dist_1_to_point = np.linalg.norm(point1 - point)
    dist_point_to_2 = np.linalg.norm(point - point2)
    dist_1_to_2 = np.linalg.norm(point1 - point2)
    
    # Check if point lies approximately on the line between point1 and point2
    # The sum of distances should approximately equal the direct distance
    return abs((dist_1_to_point + dist_point_to_2) - dist_1_to_2) < tolerance

# Enhanced main processing function
def process_wmh_masks_with_ventricles(wmh, v_wmh, ventricle_masks, voxel_size, min_area_pixels=10, adjacency_threshold=2, alignment_threshold=3):
    """
    Process WMH masks considering ventricle context for each slice.
    
    Parameters:
    -----------
    wmh : numpy.ndarray
        3D abnormal WMH mask (H, W, D)
    v_wmh : numpy.ndarray  
        3D normal/juxtaventricular WMH mask (H, W, D)
    ventricle_masks : numpy.ndarray
        3D ventricle mask (H, W, D)
    voxel_size : tuple
        Voxel spacing (height, width, depth) in mm
        
    Returns:
    --------
    updated_wmh : numpy.ndarray
        Updated abnormal WMH mask
    updated_v_wmh : numpy.ndarray
        Updated normal WMH mask
    """
    
    updated_wmh = wmh.copy()
    updated_v_wmh = v_wmh.copy()
        
    # Process each slice
    for slice_idx in range(wmh.shape[-1]):
        print(f"Processing slice {slice_idx + 1}/{wmh.shape[-1]}")
        
        wmh_slice = wmh[..., slice_idx]
        v_wmh_slice = v_wmh[..., slice_idx]
        ventricle_slice = ventricle_masks[..., slice_idx]
        
        # Find pixels to move using enhanced algorithm
        pixels_to_move = enhanced_adjacency_finder_with_ventricles(
            wmh_slice, v_wmh_slice, ventricle_slice,
            min_area=min_area_pixels,
            adjacency_threshold=adjacency_threshold,  # 1mm threshold
            alignment_threshold=alignment_threshold,  # 
            voxel_size=voxel_size[:2]  # Only height and width for 2D processing
        )
        
        # Update masks
        updated_v_wmh[..., slice_idx] = updated_v_wmh[..., slice_idx] & ~pixels_to_move
        updated_wmh[..., slice_idx] = updated_wmh[..., slice_idx] | pixels_to_move
        
        # Print statistics for this slice
        moved_pixels = np.sum(pixels_to_move)
        if moved_pixels > 0:
            print(f"  Slice {slice_idx}: Moved {moved_pixels} pixels from normal to abnormal WMH")
    
    return updated_wmh, updated_v_wmh


# %%
def wmh_vent_distance(
    obj_contour: 'np.ndarray',
    vent_image: 'np.ndarray',
    pixel_size: list
) -> float:
    """
    Calculate the minimum distance between a single WMH object contour and the ventricle contours in a given slice.

    Parameters:
        obj_contour (np.ndarray): Contour of the WMH object (single contour).
        vent_image (np.ndarray): Binary image containing all ventricle contours in the slice.
        pixel_size (list): A two-element list or array specifying the physical size of each pixel (e.g., [width, height] in mm).
    
    Returns:
        float: The minimum distance between the WMH contour and the nearest ventricle contour, in physical units.
    """
    vent_contour, _ = cv2.findContours(vent_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if len(vent_contour) != 0:

        # obj_contour is an obj of WMHs in a slice.
        # vent_contour is ventricles in a corresponding slice.

        # Measuring minimum distance between the object and ventricular system
        # Initialize the minimum distance as a large value
        min_distance = float('inf')
        min_distance_cnt = float('inf')

        # Calculate the centroid of the obj
        M = cv2.moments(obj_contour)


        # Calculate the centroid coordinates (center of mass)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # there might be only one or two points in the obj_contour:
            cx = obj_contour[0][0][0]
            cy = obj_contour[0][0][1]

        # Print the centroid coordinates
        # print(f"Centroid coordinates (x, y): ({cx}, {cy})")

        # First, min distance due to the center of obj:
        # Iterate through all pairs of points in the two or more contours:
        for vent_cont in vent_contour:
            for point1 in vent_cont:

                distance = np.linalg.norm(point1 - [cx, cy])
                # print(distance)

                # Update the minimum distance if a smaller distance is found
                if distance < min_distance_cnt:
                    min_distance_cnt = distance

        # Second, min distance due to boundaries:
        # Iterate through all pairs of points in the two or more contours:
        for vent_cont in vent_contour:
            for point1 in vent_cont:
                for point2 in obj_contour:
                    # Calculate the Euclidean distance between the two points
                    distance = np.linalg.norm(point1 - point2)
                    # print(distance)

                    # Update the minimum distance if a smaller distance is found
                    if distance < min_distance:
                        min_distance = distance

        min_distance = np.round(min_distance * pixel_size, 3)
        min_distance_cnt = np.round(min_distance_cnt * pixel_size, 3)
        # Print the minimum distance
        # print(f"Minimum distance between the edges of the vents and the center of the object: {min_distance_cnt}")
        # print(f"Minimum distance between the edges of the vents and the object: {min_distance}")
    else:
        min_distance = min_distance_cnt = 1000   # a large number!
        # Print the minimum distance
        print(f'\nThere is no seen ventricle in the slice, so min_distance cannot be defined.'
              f'  (it is set to a large value : {min_distance})')

    return min_distance_cnt, min_distance, vent_contour


# %%
def wmh_csf_distance(
    obj_contour: 'np.ndarray',
    csf_image: 'np.ndarray',
    pixel_size: list
) -> float:
    """
    Calculate the minimum distance between a single WMH object contour and the cerebrospinal fluids (CSF) contours in a given slice.

    Parameters:
        obj_contour (np.ndarray): Contour of the WMH object (single contour).
        csf_image (np.ndarray): Binary image containing all cerebrospinal fluids (CSF) regions in the slice.
        pixel_size (list): A two-element list or array specifying the physical size of each pixel (e.g., [width, height] in mm).
    
    Returns:
        float: The minimum distance between the WMH contour and the nearest CSF contour, in physical units.
    """
    # First of all, we should check whether there is an overlapping between obj (our its contour) and GM map or not!
    check = 1
    csf_cont = []
    for point in obj_contour:

        if csf_image[point[0][0], point[0][1]] == 1:
            # object is in the csf map:
            # check = 0
            break
    if check == 1:
        min_csf_distance_cnt, min_csf_distance, csf_cont = wmh_vent_distance(obj_contour, csf_image, pixel_size)

    else:
        # Print the minimum distance
        min_csf_distance_cnt = min_csf_distance = 1000
        print(f'\nThere is no distinction between the object and CSF region, so min_distance cannot be defined.'
              f'  (it is set to a zero value : {min_csf_distance})')

    return min_csf_distance_cnt, min_csf_distance, csf_cont


# %%
def obj_categorize(obj_area, _v, _v_cnt, _g, vent_rule=10, gm_rule=(5, 5)):
    # based on the rules:
    # decide the category:

    code = 0
    if _v <= .5 * vent_rule:
        # the WMH is "Periventricular"
        # print('peri')
        code = 1

    elif _v_cnt <= vent_rule:
        # the WMH is "Periventricular"
        # print('peri')
        code = 1

    elif _g <= gm_rule[0] and obj_area <= np.pi * (gm_rule[1] ** 2) / 4:
        # the WMH is "Juxtacortical"
        # print('juxt')
        code = 3

    else:
        # the WMH is "Paraventricular"
        # print('para')
        code = 2

    return code


# %%
def obj_analyzer(
    bw_image: 'np.ndarray',
    vent_image: 'np.ndarray', 
    c_mask: 'np.ndarray',
    voxel_size: float, 
    vent_rule: str, 
    gm_rule: str
    ) -> dict:
    """
    Analyze objects in medical images based on given rules and parameters.
    
    Parameters:
        bw_image (np.ndarray): Binary white matter hyperintensity (WMH) slice image.
        vent_image (np.ndarray): Binary ventricle slice image.
        c_mask (np.ndarray): Binary CSF slice mask.
        voxel_size (float): Size of a single voxel in the image (in mm³).
        vent_rule (str): Rule for ventricle analysis.
        gm_rule (str): Rule for gray matter analysis (e.g., "distance" or "diameter").
    
    Returns:
        dict: A dictionary containing analysis results.
    """

    ## Analyzing the WMH
    contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # the number of found objects
    number = len(contours)
    print(f"                                          Number of Objects: {number}")

    # Prepare to go in a loop through each contour
    areas = []
    areas_code = []

    peri_mask = np.zeros(bw_image.shape)
    para_mask = np.zeros(bw_image.shape)
    juxt_mask = np.zeros(bw_image.shape)

    if len(contours) == 0:
        print('\nThere is no object in the given image/slice!\n')
        pass

    else:
        for contour in contours:

            print("---------------------------------------------------------------------------------------------------")
            # Calculate the area & intensity index of the contour

            # Create a blank mask for the current contour
            blank = np.zeros((bw_image.shape[0], bw_image.shape[1], 3), dtype=np.uint8)
            cv2.drawContours(blank, [contour], -1, 1, -1)  # Fill the contour with value 1
            blank = np.where(blank > 0, 1, 0).astype(np.uint8)[..., 0]  # Create binary mask

            # Calculate the area in real-world units (e.g., mm²)
            area = np.round(np.sum(blank) * voxel_size[0] * voxel_size[1], 3)
            areas.append(area)

            # Compute Ventricular distance for the object of study:
            min_v_distance_cnt, min_v_distance, vent_cont = wmh_vent_distance(contour, vent_image, voxel_size[0])

            # Compute CSF distance for the object of study:
            min_csf_distance_cnt, min_csf_distance, csf_cont = wmh_csf_distance(contour, c_mask, voxel_size[0])

            # Print the area, csf_distance, vent_distance, for each object
            print(f"Object Area: {area}")
            print(f"Distance from the Ventricles: shortest ==> {min_v_distance} , centroid ==> {min_v_distance_cnt}")
            print(f"Distance from the CSF: shortest        ==> {min_csf_distance} , centroid ==> {min_csf_distance_cnt}")

            # Producing new image
            # color decision (type of the WMH):
            code = obj_categorize(
                area, min_v_distance, min_v_distance_cnt, min_csf_distance, vent_rule, gm_rule
            )
            areas_code.append(code)
            if code > 0:
                if code == 1:
                    # it is a periventricular WMH
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(peri_mask, [contour], -1, 1, -1)
                elif code == 2:
                    # it is a paraventricular WMH
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(para_mask, [contour], -1, 1, -1)
                elif code == 3:
                    # it is a juxtacortical WMH
                    # adding the corresponding object to the premier blank mask
                    cv2.drawContours(juxt_mask, [contour], -1, 1, -1)

    return (peri_mask, para_mask, juxt_mask,
            areas_code, contours, 
            number, areas)


# %%
def brain_mask_new(data_img):

    mask_img = np.zeros((data_img.shape), dtype=np.uint8)
    pad_width = 28

    brain_centers = np.zeros((1, 2, data_img.shape[2]))
    brain_axes = np.zeros((1, 2, data_img.shape[2]))

    area_e = 0
    for h in range(data_img.shape[2]):

        # Load the grayscale MRI image
        image = 255 * (data_img[..., h] / np.max(data_img[..., h]))
        image = np.pad(image,
                       pad_width=((pad_width, pad_width), (pad_width, pad_width)),
                       mode='constant',
                       constant_values=0)


        # 1. Thresholding the Image
        threshold_value = int(255 / 10)  # min_val + (np.percentile(image, 10) - min_val)
        _, initial_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert mask to boolean for morphological operations
        initial_mask_bool = initial_mask.astype(bool)

        # 2. Morphological Operations (Open/Close) using a diamond-shaped structuring element

        struct_elem = diamond(1)  # Create a diamond-shaped structuring element

        # Apply opening and closing to fill the mask
        opened_mask = binary_opening(initial_mask_bool, struct_elem)
        closed_mask = binary_closing(opened_mask, struct_elem)

        dilated_mask = dilation(closed_mask, diamond(1))

        struct_elem = diamond(4)  # Create a diamond-shaped structuring element

        # Apply opening and closing to fill the mask
        # opened_mask = binary_opening(initial_mask_bool, struct_elem)
        closed_mask = binary_closing(dilated_mask, struct_elem)

        # Convert the processed mask back to uint8
        filled_mask = (closed_mask * 255).astype(np.uint8)

        # # 3. Apply the Eroded Mask to the Original Image to Extract the Skull
        # skull_image = cv2.bitwise_and(image, image, mask=eroded_mask_uint8)

        # 4. Find contours in the mask
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # print(f'\t\t Contours: {len(contours)}')

            # Find the largest contour (assuming the skull is the largest object in the mask)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit an ellipse to the largest contour
            if len(largest_contour) >= 5:  # At least 5 points are needed to fit an ellipse
                ellipse = cv2.fitEllipse(largest_contour)

                # Calculate the area of the ellipse
                axes = ellipse[1]
                brain_axes[..., h] = axes
                ellipse_area = np.pi * (axes[0] / 2) * (axes[1] / 2)

                # Calculate the center coordinates
                if ellipse_area > area_e:
                    # update area_e:
                    # area_e = ellipse_area

                    # save the cneters:
                    center_x, center_y = map(int, ellipse[0])
                    brain_centers[0, 0, h] = center_x  - pad_width
                    brain_centers[0, 1, h] = center_y  - pad_width

                # Create a blank image to draw the ellipse
                ellipse_image = np.zeros_like(filled_mask)

                # Draw the ellipse on the blank image
                cv2.ellipse(ellipse_image, ellipse, 255, thickness=-1)  # Filled ellipse with white color

                # 5. Erosion to Shrink the Mask by 10 Pixels
                eroded_mask = erosion(ellipse_image, diamond(10))
                eroded_mask_uint8 = (eroded_mask * 1).astype(np.uint8)

                # 6.

                # 7. Unpad the obtained eroded mask
                eroded_mask_uint8_unpad = eroded_mask_uint8[pad_width:-pad_width, pad_width:-pad_width]
                mask_img[..., h] = eroded_mask_uint8_unpad

            else:
                print("\t\tNot enough points to fit an ellipse.")
        else:
            print("\t\tNo Contours to fit an ellipse")

    return mask_img, brain_centers, brain_axes


# %% Function to apply adaptive thresholding and generate masks of potential CSF regions
def generate_adaptive_dark_masks(image_stack, block_size=65, c=25, min_area=2, darkness_threshold=80, display_sample=False):
    """
    Thresholds images in a 2D stack adaptively to extract dark regions as masks.

    Args:
        image_stack (numpy.ndarray): 3D array (H, W, N) where N is the number of images.
        block_size (int): Size of the local region for adaptive thresholding (must be odd).
        c (int): Constant to subtract from the mean or weighted mean.
        min_area (int): Minimum area of connected components to retain in the mask.
        darkness_threshold (int): Global pixel value threshold for darkness (0-255).
        display_sample (bool): If True, displays the first image and its mask.

    Returns:
        numpy.ndarray: 3D array (H, W, N) of binary masks.
    """
    masks = []

    for i in range(image_stack.shape[-1]):
        image = image_stack[..., i]

        # Ensure the image is in 8-bit format for thresholding
        image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply adaptive thresholding to segment dark regions
        adaptive_mask = cv2.adaptiveThreshold(image_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY_INV, block_size, c)

        # Combine with global darkness threshold
        global_mask = (image_8bit < darkness_threshold).astype(np.uint8) * 255
        combined_mask = cv2.bitwise_and(adaptive_mask, global_mask)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive_mask, connectivity=8)
        cleaned_mask = np.zeros_like(adaptive_mask)

        for j in range(1, num_labels):  # Skip the background label (0)
            if stats[j, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == j] = 255

        # Store the cleaned mask
        masks.append(cleaned_mask)

        # Display the first image and its mask if requested
        if display_sample: # and i == 0:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title("Adaptive Mask")
            plt.imshow(cleaned_mask, cmap='gray')
            plt.axis('off')

            plt.show()

    return np.array(masks).transpose((1, 2, 0))


# %%
def main_process(vent_wmh_prediction_image, flair_data_image, u_folder='/save_directory'):
    """Placeholder for the main process logic."""
    voxel_size = (0.9, 0.9)
    vent_wmh_masks = np.copy(vent_wmh_prediction_image)

    try:

        # % [markdown]
        # ### 4L Mask Post-processing

        # %
        # Morphologically post-processing the three masks.

        # for ventricle masks:
        vent_m = np.where(vent_wmh_masks ==1, 1, 0)

        # for WMH masks:
        wmh = np.where(vent_wmh_masks ==2, 1, 0)

        # % [markdown]
        # ## Phase 4-1: Inputs

        # % [markdown]
        # ### Load prepared data and information

        # from segmenting phase:
        vent_mask = np.copy(vent_m)
        wmh_mask = np.copy(wmh)

        # prepare CSF mask
        # brain extraction:
        masks_data, brain_cnt, brain_ax = brain_mask_new(np.expand_dims(flair_data_image, axis=-1))
        masks_data = np.squeeze(masks_data, axis=-1)
        masks_data = np.where(masks_data < 128, 0, 1).astype(np.uint8)
        # generate CSF-candidate masks:
        csf_mask = np.squeeze(generate_adaptive_dark_masks(np.expand_dims(flair_data_image, axis=-1)), axis=-1) * masks_data


        # % [markdown]
        # ## Phase 4-2: Analyzing

        # % [markdown]
        # ##### Here, we aim to recall segmented ventricles and detected WMHs of each subject in order to first, omit non-plaque WMHs, second, categorize plaque-like WHMs into JVWMH, PEWMH, PAWMH, and JCWMH groups. Eventually, we will statistically analyze them from aspects of frequency, area, penetration, and location. (& intensity!)
        #

        # %
        # Before proceeding with main analysis, we should define some specific number of slices, for all subjects...
        # ... to continue our process just with them. (to avoid extra non-interested slices that might impose further issues)
        # (assuming inferior to superior direction of slices)

        # %
        # constant values:
        vent_rule = 15              # enter max distance from ventricles allowed
        gm_rule = (6, 6)            # enter (Distance from GM mask, max Diameter of allowed objects)

        # %
        # analysis:

        # make sure the input masks are binary valued
        vent_mask = np.where(vent_mask > 0, 1, 0).astype(np.uint8)
        wmh_mask = np.where(wmh_mask > 0, 1, 0).astype(np.uint8)
        csf_mask = np.where(csf_mask > 0, 1, 0).astype(np.uint8)

        # 1. total number, area, codes:         in 4 groups  {all, PEWMH, PAWMH, JCWMH}

        print(f"\n\n                                             Image : {os.path.basename(u_folder)}\n")

        (peri_found, para_found, juxt_found,
        areas_code, contours_wmh,
        number_wmh, areas_wmh
        ) = obj_analyzer(
            wmh_mask, vent_mask, c_mask=csf_mask,
            voxel_size=voxel_size, vent_rule=vent_rule, gm_rule=gm_rule
            )

        # %
        # Save the results for further analysis:

        # Define color codes (RGB values)
        color_codes = {
            'red': (255, 0, 0),  # peri_found
            'orange': (255, 165, 0),  # para_found
            'yellow': (255, 255, 0),  # juxt_found
            'blue': (0, 0, 255),  # vent_mask
        }

        # Initialize RGB mask
        rgb_mask = np.zeros((pred_image.shape[0], pred_image.shape[1], 3), dtype=np.uint8)

        # Apply colors to the RGB mask
        # Order matters - later assignments will overwrite earlier ones where masks overlap

        # Blue for ventricles
        rgb_mask[vent_mask == 1] = color_codes['blue']

        # Red for periventricular
        rgb_mask[peri_found == 1] = color_codes['red']

        # Orange for paraventricular
        rgb_mask[para_found == 1] = color_codes['orange']

        # Yellow for juxtacortical
        rgb_mask[juxt_found == 1] = color_codes['yellow']

        # Save the RGB mask using skimage
        skimage.io.imsave(u_folder.replace('.npy', 'tif'), rgb_mask)
        np.save(u_folder, rgb_mask)

        # Save the analysis results in a single file
        # Create a dictionary to store all results
        results_data = {
            'areas_code': areas_code,
            'number_wmh': number_wmh,
            'areas_wmh': areas_wmh,
            'color_codes': color_codes,  # Include color mapping for reference
            'mask_info': {
                'peri_count': np.sum(peri_found),
                'para_count': np.sum(para_found),
                'juxt_count': np.sum(juxt_found),
                'vent_count': np.sum(vent_mask)
            }
        }

        # Create results filename (same base name as image file)
        results_filename = os.path.splitext(u_folder)[0] + '_results.pkl'

        # Save as pickle file (handles different shapes/types well)
        with open(results_filename, 'wb') as f:
            pickle.dump(results_data, f)

        print(f"RGB mask saved to: {result_file_path}")
        print(f"Analysis results saved to: {results_filename}")

        # # Alternative: Save as .npz file (numpy format)
        # results_filename_npz = os.path.splitext(result_file_path)[0] + '_results.npz'
        # np.savez_compressed(results_filename_npz,
        #                     areas_code=areas_code,
        #                     number_wmh=number_wmh,
        #                     areas_wmh=areas_wmh,
        #                     peri_count=np.sum(peri_found),
        #                     para_count=np.sum(para_found),
        #                     juxt_count=np.sum(juxt_found),
        #                     vent_count=np.sum(vent_mask))
        #
        # print(f"Analysis results also saved to: {results_filename_npz}")

        # To load the pickle file later:
        # with open(results_filename, 'rb') as f:
        #     loaded_results = pickle.load(f)

        # To load the npz file later:
        # loaded_data = np.load(results_filename_npz)
        # areas_code = loaded_data['areas_code']
        # number_wmh = loaded_data['number_wmh']
        # areas_wmh = loaded_data['areas_wmh']


        return

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        return


if __name__ == "__main__":
    # Directory of ventricle_WMH prediction image files in actual dimensions
    predictions_dir = r"C:\Users\Mehdi\Documents\Thesis\Papers\ours\Paper#Stats\transformed_predictions\numpy_transformed_predictions"

    # Directory of preprocessed FLAIR data in actual dimensions
    flair_dir = r"C:\Users\Mehdi\Documents\Thesis\Papers\ours\Paper#Stats\MS_transformed_images"

    # Directory of resulting masks
    result_dir = r"C:\Users\Mehdi\Documents\Thesis\Papers\ours\Paper#Stats\MS_final_np_masks"
    os.makedirs(result_dir, exist_ok=True)

    pred_images_files = [f for f in os.listdir(predictions_dir) if f.endswith('.npy')]
    flair_images_files = [f for f in os.listdir(flair_dir) if f.endswith('.png')]

    for pred_image_file in pred_images_files:
        # Load prediction file
        pred_image_path = os.path.join(predictions_dir, pred_image_file)
        p_id = pred_image_file.split('_')[0]
        slice_n = pred_image_file.split('_')[-4]
        # pred_image = skimage.io.imread(pred_image_path)
        pred_image = np.load(pred_image_path)

        # Find corresponding flair image from the flair directory
        for flair_image_file in flair_images_files:
            if p_id in flair_image_file:
                if flair_image_file.split('_')[-2] == slice_n:
                    flair_image_path = os.path.join(flair_dir, flair_image_file)
                    break

        # Load FLAIR file
        flair_image = skimage.io.imread(flair_image_path)

        # Make result file path
        result_file_path = os.path.join(result_dir, pred_image_file.replace('.npy', '_final.npy'))

        # Call the core processing
        main_process(pred_image, flair_image, result_file_path)






