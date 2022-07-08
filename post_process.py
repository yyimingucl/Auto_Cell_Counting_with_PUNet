#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yiming Yang
# Date: 08/07/2022
# Email: y.yang2@napier.ac.uk
# Description: post_processing for generated binary mask and calculate the counting numbers for cells


import numpy as np
from skimage import measure
from scipy import ndimage
from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.feature import peak_local_max
import skimage.segmentation as ski_seg



def mask_post_processing(pred_mask, area_threshold=5, min_obj_size=5, foot=3):
    """ 
    Marked Watershed Algorithm for post-processing the generated binary mask 
    1. fill the small holes 
    2. remove the small objects 
    3. compute the euclidean distance from peak to background and make transformation
    4. locate local maxima 
    5. define the marks for watershed by local maxima 
    6. marked watershed algorithm   

    Args:
        pred_mask (np.array): prediceted binary mask
        area_threshold (int, optional): maximum hole size to be filled. Defaults to 5.
        min_obj_size (int, optional): maximum object size to be removed. Defaults to 5.
        foot (int, optional): define the maximum filter with size (foot, foot). Defaults to 3.

    Returns:
        post_processed mask (np.array)
    """
    

    # Find object in predicted image
    thresh_mask = pred_mask.astype(bool)
    labels_pred, __ = ndimage.label(thresh_mask)
    processed = remove_small_holes(labels_pred, area_threshold=area_threshold, connectivity=1)
                                #    in_place=True)
    processed = remove_small_objects(
        processed, min_size=min_obj_size, connectivity=1)#, in_place=True)
    labels_bool = processed.astype(bool)

    distance = ndimage.distance_transform_edt(processed)
    max_dist = 0.9*np.max(distance)
    maxi = ndimage.maximum_filter(distance, size=max_dist, mode='constant')
    local_maxi = peak_local_max(maxi, indices=False, footprint=np.ones((foot, foot)),
                                exclude_border=False,
                                labels=labels_bool)

    local_maxi = remove_small_objects(
        local_maxi, min_size=min_obj_size, connectivity=1)#, in_place=False)
    markers = ndimage.label(local_maxi)[0]
    labels = ski_seg.watershed(-distance, markers, mask=labels_bool,
                       compactness=1, watershed_line=True)

    return (labels.astype("uint8")*255)

def get_single_class_counting_number(pred_mask):
    
    # get the counting numbers for single class problem

    post_pred = mask_post_processing(pred_mask)
    _, pred_count = ndimage.label(post_pred)
    return pred_count
