#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 12:23:59 2020

@author: sadhana-ravikumar
"""

import numpy as np
import SimpleITK as sitk
import config.runconfig as runconfig 
from torch.utils.data import Dataset
import torch
import pandas as pd
import utils.patch_gen as p
import random
import nibabel as nib
import csv
import os.path as osp

c = runconfig.Config_Run()

#Writes a patch (imaeg and seg) to file. Saves control point in csv
def write_patch_to_file(image, seg, coordmaps, sample, cpt, idx):
    
        img_id = sample['id']
        phase = sample['type']
        data_list = []
        
        # save image
        img_filename = osp.join(c.patchesDataDir, phase + "_img_"+ str(img_id) + '_idx_' + str(idx) + ".nii.gz")
        image = np.rollaxis(image, 0, 4)
        nib.save(nib.Nifti1Image(image, sample['affine']), img_filename)
        data_list.append(img_filename)
        
        # save segmentation
        seg_filename = osp.join(c.patchesDataDir , phase + "_seg_"+ str(img_id) +  '_idx_' +str(idx) + ".nii.gz" )
        seg = np.rollaxis(seg, 0, 4)
        nib.save(nib.Nifti1Image(seg, sample['affine']), seg_filename)
        data_list.append(seg_filename)
        
        # save coordinate maps
        coordmaps_filename = osp.join(c.patchesDataDir , phase + "_coordmaps_"+ str(img_id) +  '_idx_' +str(idx) + ".nii.gz" )
        coordmaps = np.rollaxis(coordmaps, 0, 4)
        nib.save(nib.Nifti1Image(coordmaps, sample['affine']), coordmaps_filename)
        data_list.append(coordmaps_filename)
            
        #save filenames and corresponding control point in csv
        data_list.append(img_id)
        data_list.append(cpt)
        data_list.append(phase)
        if phase == 'train':
            outcsv = c.train_patch_csv
        elif phase == 'val':
            outcsv = c.val_patch_csv
        else:
            outcsv = c.test_patch_csv
        with open(outcsv, 'a') as f:
            wr = csv.writer(f)
            wr.writerow(data_list)

# def standardize_image(image_np):
    
#     out_image = image_np
#     for channel in range(image_np.shape[0]):
        
#         image_channel = image_np[channel, ...]
#         image_voxels = image_channel[image_channel>0] # Get rid of the background
#         out_image[channel, ...] = (image_channel - np.mean(image_voxels)) / np.std(image_voxels)
       
#     return out_image
        
def get_patches(image, label, coordmaps, sample, num_pos = 100, num_neg = 100, all_patches=False, patch_shape= (48,48,48), spacing=(24,24,24), start_idx = 0):
    """
    get image, and label patches
    Default: returns one randomly selected patch of the given size, else
    returns a list of all the patches.
    """
    image_shape = np.shape(image)
    cn_size = image_shape[0]
    sg_size = image_shape[1]
    cr_size = image_shape[2]
    ax_size = image_shape[3]

    if not all_patches:
        idx_pos = np.stack(np.where(label[0, ...] > 0))
    
        # Only include points not near boundary
        #sg_idx = np.where(((patch_shape[0]/2) < idx_pos[0]) & (idx_pos[0] < (sg_size - (patch_shape[0]/2))))
        #idx_pos = idx_pos[:,sg_idx[0]]
        #cr_idx = np.where(((patch_shape[1]/2) < idx_pos[1]) & (idx_pos[1] < (cr_size - (patch_shape[1]/2))))
        #idx_pos = idx_pos[:, cr_idx[0]]
        #ax_idx = np.where(((patch_shape[2]/2) < idx_pos[2]) & (idx_pos[2] < (ax_size - (patch_shape[2]/2))))
        #idx_pos = idx_pos[:, ax_idx[0]]
        
        idx_rand = np.random.choice(idx_pos[0].shape[0], num_pos, replace = False)
        cpts_pos_sampled = idx_pos[:, idx_rand] 
           
        image_patch_list = []
        label_patch_list = []
        coordmaps_patch_list = []
        for i in range(num_pos):
            idx1_sg = cpts_pos_sampled[0][i] - int(patch_shape[0]/2)
            idx1_cr = cpts_pos_sampled[1][i] - int(patch_shape[1]/2)
            idx1_ax = cpts_pos_sampled[2][i] - int(patch_shape[2]/2)
        
            idx2_sg = idx1_sg + patch_shape[0]
            idx2_cr = idx1_cr + patch_shape[1]
            idx2_ax = idx1_ax + patch_shape[2]
        
            image_patch_orig = image[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            image_patch = p.standardize_image(image_patch_orig)
            #image_patch = p.Normalize(image_patch)
            label_patch = label[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            coordmaps_patch = coordmaps[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
               
            image_patch_list.append(image_patch)
            label_patch_list.append(label_patch)
            coordmaps_patch_list.append(coordmaps_patch)
            
            #Write patch/image and control points to csv and save image
            write_patch_to_file(image_patch, label_patch, coordmaps_patch, sample, cpts_pos_sampled[:,i], start_idx + i)
            
        # For negative points
        idx_neg = np.stack(np.where(label[0, ...]==0), axis = 0)
        
        # Only include points not near boundary
        sg_idx = np.where(((patch_shape[0]/2) < idx_pos[0]) & (idx_pos[0] < (sg_size - (patch_shape[0]/2))))
        idx_neg = idx_neg[:,sg_idx[0]]
        cr_idx = np.where(((patch_shape[1]/2) < idx_pos[1]) & (idx_pos[1] < (cr_size - (patch_shape[1]/2))))
        idx_neg = idx_neg[:, cr_idx[0]]
        ax_idx = np.where(((patch_shape[2]/2) < idx_pos[2]) & (idx_pos[2] < (ax_size - (patch_shape[2]/2))))
        idx_neg = idx_neg[:, ax_idx[0]]
        
        idx_rand = np.random.choice(idx_neg[0].shape[0], num_neg, replace = False)
        cpts_neg_sampled = idx_neg[:, idx_rand] 
        
        for i in range(num_neg):
            idx1_sg = cpts_pos_sampled[0][i] - int(patch_shape[0]/2)
            idx1_cr = cpts_pos_sampled[1][i] - int(patch_shape[1]/2)
            idx1_ax = cpts_pos_sampled[2][i] - int(patch_shape[2]/2)
        
            idx2_sg = idx1_sg + patch_shape[0]
            idx2_cr = idx1_cr + patch_shape[1]
            idx2_ax = idx1_ax + patch_shape[2]
        
            image_patch_orig = image[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            image_patch = p.standardize_image(image_patch_orig)
            #image_patch = p.Normalize(image_patch)
            label_patch = label[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            coordmaps_patch = coordmaps[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
                
            image_patch_list.append(image_patch)
            label_patch_list.append(label_patch)
            coordmaps_patch_list.append(coordmaps_patch)
            
            #Write patch/image and control points to csv and save image
            write_patch_to_file(image_patch, label_patch, coordmaps_patch, sample, cpts_pos_sampled[:,i], start_idx + num_pos + i)
           
        cpts = np.concatenate((cpts_pos_sampled, cpts_neg_sampled), axis = 1)
        
        return image_patch_list, label_patch_list, coordmaps_patch_list, cpts, start_idx + num_pos + i

    else:
        
        idx = p.grid_center_points(image.shape[1:], spacing)
       
        # Only include points not near boundary
        sg_idx = np.where(((patch_shape[0]/2) < idx[0]) & (idx[0] < (sg_size - (patch_shape[0]/2))))
        idx = idx[:,sg_idx[0]]
        cr_idx = np.where(((patch_shape[1]/2) < idx[1]) & (idx[1] < (cr_size - (patch_shape[1]/2))))
        idx = idx[:, cr_idx[0]]
        ax_idx = np.where(((patch_shape[2]/2) < idx[2]) & (idx[2] < (ax_size - (patch_shape[2]/2))))
        idx = idx[:, ax_idx[0]]
        
        image_patch_list = []
        label_patch_list = []
        coordmaps_patch_list = []
        
        for i in range(idx.shape[1]):
            
            idx1_sg = idx[0][i] - int(patch_shape[0]/2)
            idx1_cr = idx[1][i] - int(patch_shape[1]/2)
            idx1_ax = idx[2][i] - int(patch_shape[2]/2)
       
            idx2_sg = idx1_sg + patch_shape[0]
            idx2_cr = idx1_cr + patch_shape[1]
            idx2_ax = idx1_ax + patch_shape[2]
        
            image_patch_orig = image[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            image_patch = p.standardize_image(image_patch_orig)
            #image_patch = p.Normalize(image_patch)
            label_patch = label[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
            coordmaps_patch = coordmaps[:, idx1_sg:idx2_sg, idx1_cr:idx2_cr, idx1_ax:idx2_ax]
               
            image_patch_list.append(image_patch)
            label_patch_list.append(label_patch)
            coordmaps_patch_list.append(coordmaps_patch)
        
        return image_patch_list, label_patch_list, coordmaps_patch_list, idx, len(image_patch_list)
    
    
def generate_patches(sample, config):
   
    img = sample['image']
    seg = sample['seg']
    coordmaps = sample['coordmaps']
    phase = sample['type']
    
    # load atlas info
    with_atlas = config.get('with_atlas', False)
    if with_atlas:
        natlas = seg.shape[0] - 1
        nmodality = int(img.shape[0]/natlas)
        num_atlas_aug = config['num_atlas_aug']
        num_sel_atlas = config['num_sel_atlas']
    else:
        natlas = 0
        nmodality = int(img.shape[0])
        num_atlas_aug = 1
        num_sel_atlas = 0
    
    # determine number of positive and negative samples
    if phase == 'train':
        num_pos = config['num_pos_train']
        num_neg = config['num_neg_train']
    else:
        num_pos = config['num_pos_val']
        num_neg = config['num_neg_val']
    
    # standardize and normalize images
    #img_std = p.standardize_image(img)
    #img_norm = p.Normalize(img_std)
    img_norm = img
        
    if phase != 'test':
        all_patches = False
        spacing = None
        
        # perform sampling if needed
        img_patches = []
        seg_patches = []
        coordmaps_patches = []
        cpts = None
        start_idx = 0
        
        # perform atlas augmentation if necessary
        for s in range(num_atlas_aug):
        
            # sample atlases
            if with_atlas:
                if num_sel_atlas <= natlas:
                    rnd_idx = random.sample(range(1,natlas+1), k = num_sel_atlas)
                else:
                    rnd_idx1 = random.sample(range(1,natlas+1), k = natlas)
                    rnd_idx2 = np.random.choice(range(1,natlas+1), (num_sel_atlas-natlas))
                    rnd_idx = [rnd_idx1, rnd_idx2]

                # select segmentations
                seg_rnd_idx = np.insert(rnd_idx, 0, 0)
                seg_sel = seg[seg_rnd_idx, ...]
                
                # select corresponding multimodal images
                img_rnd_idx = None
                for t in range(seg_rnd_idx.shape[0]):
                    begin_idx = seg_rnd_idx[t] * nmodality
                    idxrange = np.arange(begin_idx, begin_idx+nmodality)
                    if img_rnd_idx is None:
                        img_rnd_idx = idxrange
                    else:
                        img_rnd_idx = np.concatenate((img_rnd_idx, idxrange))
                img_sel = img_norm[img_rnd_idx, ...]
            else:
                seg_sel = seg
                img_sel = img_norm
        
            # Pad test image to take care of boundary condition
            pad_size = np.ceil((np.array(config['patch_size']) - 1) / 2).astype(np.int32)
            img_sel = np.pad(img_sel, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)
            seg_sel = np.pad(seg_sel, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)
            coordmaps_sel = np.pad(coordmaps, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)


            #Returns center points of patches - useful when reconstructing test image
            img_patches_sample, seg_patches_sample, coordmaps_patches_sample, cpts_sample, end_idx = get_patches(img_sel, seg_sel, coordmaps_sel, sample,
                                patch_shape = config['patch_size'], num_pos = num_pos, num_neg = num_neg, 
                                all_patches = all_patches, spacing = spacing, start_idx = start_idx)
            img_patches = img_patches + img_patches_sample
            seg_patches = seg_patches + seg_patches_sample
            coordmaps_patches = coordmaps_patches + coordmaps_patches_sample
            if cpts is None:
                cpts = cpts_sample
            else:
                cpts = np.concatenate((cpts, cpts_sample), axis = 1)
            start_idx = end_idx + 1
            
    else:
        
        # in the test phase, no need to sample atlases
        all_patches = True
        # Pad test image to take care of boundary condition
        pad_size = np.ceil((np.array(config['test_patch_size']) - 1) / 2).astype(np.int32)
        #Need to patch image and segmentation
        img_norm = np.pad(img_norm, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)
        seg = np.pad(seg, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)
        coordmaps = np.pad(coordmaps, ((0,0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])), mode = "constant", constant_values = 0)
    
    
        #Returns center points of patches - useful when reconstructing test image
        img_patches, seg_patches, coordmaps_patches, cpts, end_idx = get_patches(img_norm, seg, coordmaps, sample, patch_shape = config['test_patch_size'], num_pos = num_pos, num_neg = num_neg, all_patches = all_patches, spacing = config['test_patch_spacing'], start_idx = 0)

    
    # augmentation if needed in training
    num_aug = config['num_aug']
    perc_aug = config['perc_aug']
    num_patch = len(img_patches)
    num_patch_idx = num_patch
    if phase == 'train' and num_aug > 0 and perc_aug > 0:
        
        #print("Performing augmentation")
        
        for aug in range(num_aug):
        
            #Apply the transformation to a random subset of patches
            print("Performing augmentation: Apply the transformation to a random subset of patches.")
            rnd_idx = random.sample(range(0,num_patch), k = int(np.ceil(num_patch*perc_aug)))

            for r in range(len(rnd_idx)):
                i = rnd_idx[r]
                img_elastic, seg_elastic, coordmaps_elastic = p.elastic_deformation(img_patches[i], seg_patches[i], coordmaps_patches[i])
                
                img_patches.append(img_elastic)
                seg_patches.append(seg_elastic)
                coordmaps_patches.append(coordmaps_elastic)
                cpts = np.concatenate((cpts, cpts[:,i:i+1]), axis = 1)

                #Write patch/image and control points to csv and save image
                write_patch_to_file(img_elastic, seg_elastic, coordmaps_elastic, sample, cpts[:,i], num_patch_idx)
                num_patch_idx += 1
                
    return img_patches, seg_patches, coordmaps_patches, cpts
    
        
class GeneratePatches(Dataset):
    
    def __init__(self, sample, config):
        
        self.sample = sample  
        self.phase = sample['type']
        self.config = config
        self.affine = sample['affine']
        
        # sample patches
        img_patches, seg_patches, coordmaps_patches, cpts = generate_patches(sample, config)
        
        # pass the output to the class
        self.image_patches = img_patches
        self.seg_patches = seg_patches
        self.coordmaps_patches = coordmaps_patches
        self.cpts = cpts
    
             
    def __len__(self):
        return len(self.image_patches)
        
    def __getitem__(self, idx):
        sample = {'image':self.image_patches[idx], 'seg':self.seg_patches[idx], 'coordmaps': self.coordmaps_patches[idx], 'cpt':self.cpts[:,idx]}
        return sample