#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:12:14 2020

@author: sadhana-ravikumar
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter

#This function define standardization to image
def standardize_image(image_np):
    
    out_image = None
    for channel in range(image_np.shape[0]):
        
        image_channel = image_np[channel, ...]
        image_voxels = image_channel[image_channel>0] # Get rid of the background
        tmpimg = (image_channel - np.mean(image_voxels)) / np.std(image_voxels)
        tmpimg = tmpimg[None, ...]
        if out_image is None:
            out_image = tmpimg
        else:
            out_image = np.concatenate((out_image, tmpimg), axis = 0)
       
    return out_image

       
# This function generate center points in order of image. Just to keep the API consistent
def grid_center_points(shape, space):
    x = np.arange(start = 0, stop = shape[0], step = space[0] )
    y = np.arange(start = 0, stop = shape[1], step = space[1] )
    z = np.arange(start = 0, stop = shape[2], step = space[2])
    x_t, y_t, z_t = np.meshgrid(x, y, z, indexing = "ij")
    
    idx = np.stack([x_t.flatten(), y_t.flatten(), z_t.flatten()], axis = 0)
    
    return idx


def Normalize(image, min_value=0, max_value=1):
    """
    change the intensity range
    """
    value_range = max_value - min_value
    normalized_image = image
    for channel in range(image.shape[0]):
        tmpimg = image[channel, ...];
        tmpimg = (tmpimg - np.min(tmpimg)) * (value_range) / (np.max(tmpimg) - np.min(tmpimg))
        normalized_image[channel, ...] = tmpimg + min_value
    return normalized_image
         

def elastic_deformation(img, seg = None, coordmaps = None, alpha=15, sigma=3):
    """
    Elastic deformation of 3D or 4D images on a pixelwise basis
    X: image
    Y: segmentation of the image
    alpha = scaling factor the deformation
    sigma = smooting factor
    inspired by: https://gist.github.com/fmder/e28813c1e8721830ff9c which inspired imgaug through https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a
    based on [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    First a random displacement field (sampled from a gaussian distribution) is created,
    it's then convolved with a gaussian standard deviation, σ determines the field : very small if σ is large,
        like a completely random field if σ is small,
        looks like elastic deformation with σ the elastic coefficent for values in between.
    Then the field is added to an array of coordinates, which is then mapped to the original image.
    """
    
    num_img_channel = img.shape[0]
    img_shape = img.shape[1:]
    
    if seg is not None:
        num_seg_channel = seg.shape[0]
        seg_shape = seg.shape[1:]
    
        if seg_shape != img_shape and seg is not None:
            raise ValueError("Image and segmentation dimension do not match.")
       
    if coordmaps is not None:
        num_coordmaps_channel = coordmaps.shape[0]
        coordmaps_shape = coordmaps.shape[1:]
    
        if coordmaps_shape != img_shape and coordmaps is not None:
            raise ValueError("Image and coordinate maps dimension do not match.")
    
    dx = gaussian_filter(np.random.randn(*img_shape), sigma, mode="constant", cval=0) * alpha #originally with random_state.rand * 2 - 1
    dy = gaussian_filter(np.random.randn(*img_shape), sigma, mode="constant", cval=0) * alpha
    if len(img_shape)==2:
        x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')
        indices = x+dx, y+dy

    elif len(img_shape)==3:
        dz = gaussian_filter(np.random.randn(*img_shape), sigma, mode="constant", cval=0) * alpha
        x, y, z = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), np.arange(img_shape[2]), indexing='ij')
        indices = x+dx, y+dy, z+dz
    
    else:
        raise ValueError("can't deform because the image is not either 3D or 4D")

    # resample the images
    outimg = None
    for channel in range(num_img_channel):
        tmpimg = img[channel, :, :, :]
        tmpimg = map_coordinates(tmpimg, indices, order=3).reshape(img_shape)
        tmpimg = tmpimg[None, ...]
        if outimg is None:
            outimg = tmpimg
        else:
            outimg = np.concatenate((outimg, tmpimg), axis = 0)
        
    # resample the segmentation
    outseg = None
    if seg is not None:
        for channel in range(num_seg_channel):
            tmpseg = seg[channel, :, :, :]
            tmpseg = map_coordinates(tmpseg, indices, order=0).reshape(seg_shape)
            tmpseg = tmpseg[None, ...]
            if outseg is None:
                outseg = tmpseg
            else:
                outseg = np.concatenate((outseg, tmpseg), axis = 0)
                
    # resample the segmentation
    outcoordmaps = None
    if coordmaps is not None:
        for channel in range(num_coordmaps_channel):
            tmpcoordmaps = coordmaps[channel, :, :, :]
            tmpcoordmaps = map_coordinates(tmpcoordmaps, indices, order=0).reshape(coordmaps_shape)
            tmpcoordmaps = tmpcoordmaps[None, ...]
            if outcoordmaps is None:
                outcoordmaps = tmpcoordmaps
            else:
                outcoordmaps = np.concatenate((outcoordmaps, tmpcoordmaps), axis = 0)
                
    return outimg, outseg, outcoordmaps
    

    
    
# def RandomFlip(image, image_label):
#     """
#     Randomly flips the image across the given axes.
#     Note from original repo: When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
#     otherwise the models won't converge.
#     """
#     axes = (0, 1, 2)
#     image_rot = np.flip(image, axes[1])
#     label_rot = np.flip(image_label, axes[1])
#     return image_rot, label_rot