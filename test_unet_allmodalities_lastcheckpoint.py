#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:05:15 2020

@author: sadhana-ravikumar
"""

import numpy as np
import math
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
#import loss as l
import nibabel as nib
import os.path as osp
import os
import glob
import random
import sys

import utils.preprocess_data as p
from model.model import get_model
import model.utils as utils

import matplotlib.pyplot as plt

import config.runconfig as runconfig 
c = runconfig.Config_Run()

# Load and log experiment configuration
import yaml
config = yaml.safe_load(open(c.config_file, 'r'))

from utils.gen_utils import get_logger
logger = get_logger('UNet3DPredict')

DEFAULT_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
#DEFAULT_DEVICE = "cpu"
device = config.get('device', DEFAULT_DEVICE)
config['device'] = torch.device(device)
torch.cuda.set_device(1)
print(device)

def computeGeneralizedDSC(gt, seg):

    gt_seg = gt[gt > 0]
    myseg = seg[gt > 0]

    gdsc = 100*(sum(gt_seg == myseg)/ len(gt_seg))

    return gdsc


# In[2]:


# load unet model
model = get_model(config)
model_file = c.checkpointDir + '/last_checkpoint.pytorch'
logger.info(f'Loading model from {model_file}...')
utils.load_checkpoint(model_file, model, config)
logger.info(f"Sending the model to '{device}'")
model = model.to(device)
#model.eval()
print(model)


# In[4]:


# predict the targets first 
# reorganize data so that it can be read to used by data loader
reorganize_config = config['reorganize']
half_patch = np.ceil((np.array(reorganize_config['patch_size']) - 1) / 2).astype(np.int32)
loaders_config = config['loaders']
trainer_config = config['trainer']
deep_supervision = trainer_config.get('deep_supervision', False)
preserve_size = trainer_config.get('preserve_size', False)
model_config = config['model']
model_padding = model_config.get('padding', 1)
model_num_levels = model_config['num_levels']
num_class = config['model']['out_channels']
modality = loaders_config.get('modality', 'both')

# check whether atlases are included
reorganize_config = config['reorganize']
with_atlas = reorganize_config.get('with_atlas', False)

# generate folder saving information for each patch
c.force_create(c.predictDir)

# csv file
with torch.no_grad():
    #for phase in ['training', 'test']:
    for phase in ['test']:

        subjdirs = glob.glob(c.rawdataDir + "/" + phase + '/*')
        Nsubj = len(subjdirs)
        
        for nsubj in range(Nsubj):
            
            subjdir = subjdirs[nsubj]
            subjid = subjdir.split('/')[-1]


            for side in ['left', 'right']:

                # output current status
                print("Subject " + subjid + " " + side + " (" + str(nsubj+1) + "/" + str(Nsubj) + ") in " + phase + ".")
                        
                # read T1 T2 and refseg images
                TargetT1 = os.path.join(c.rawdataDir, phase, subjid, 'mprage_to_tse_native_chunk_' + side + '_resampled.nii.gz')
                TimgT1hdr = nib.load(TargetT1)
                affine = TimgT1hdr.affine
                TimgT1 = TimgT1hdr.get_fdata()
                TargetT2 = os.path.join(c.rawdataDir, phase, subjid, 'tse_native_chunk_' + side + '_resampled.nii.gz')
                TimgT2 = nib.load(TargetT2).get_fdata()
                TargetSeg = os.path.join(c.rawdataDir, phase, subjid, 'refseg_' + side + '_chunk_resampled.nii.gz')
                Tseg = nib.load(TargetSeg).get_fdata()

                #plt.imshow(TimgT2[:,36,:])
                #plt.show()
                
                # get coordinate maps
                sizex, sizey, sizez = TimgT1.shape
                xcoordmap = np.zeros(TimgT1.shape)
                for x in range(sizex):
                    xcoordmap[x,:,:] = 2*x/sizex - 1
                ycoordmap = np.zeros(TimgT1.shape)
                for y in range(sizey):
                    ycoordmap[:,y,:] = 2*y/sizey - 1
                zcoordmap = np.zeros(TimgT1.shape)
                for z in range(sizez):
                    zcoordmap[:,:,z] = 2*z/sizez - 1
                coordmaps = np.stack((xcoordmap, ycoordmap, zcoordmap), axis = 3)
                
                # array for all the images and segmentations
                if modality == 'T1':
                    img = TimgT1[..., np.newaxis]
                elif modality == 'T2':
                    img = TimgT2[..., np.newaxis]
                elif modality == "both":
                    img = np.stack((TimgT1, TimgT2), axis = 3)
                else:
                    print("Not supported modality input" + modality)
                    exit
                seg = Tseg[..., np.newaxis]

                if with_atlas:
                    for atlasdir in glob.glob(os.path.join(subjdir, 'multiatlas', 'tseg_' + side + '_train*')):

                        # output current status
                        atlasid = atlasdir.split('/')[-1]
                        idx = atlasid.split('train')[-1]
                        #print("    Sampling atlas:" + atlasid)

                        # read atlas T1 T2 and segmentation
                        if modality == 'T1' or modality == 'both':
                            AtlasT1 = os.path.join(atlasdir, 'atlas_to_native_mprage_resampled.nii.gz')
                            AimgT1 = nib.load(AtlasT1).get_fdata()
                        if modality == 'T2' or modality == 'both':
                            AtlasT2 = os.path.join(atlasdir, 'atlas_to_native_resampled.nii.gz')
                            AimgT2 = nib.load(AtlasT2).get_fdata()
                        AtlasSeg = os.path.join(atlasdir, 'atlas_to_native_segvote_resampled.nii.gz')
                        Aseg = nib.load(AtlasSeg).get_fdata()

                        # concatenate all images together
                        AimgT1 = AimgT1[..., np.newaxis]
                        AimgT2 = AimgT2[..., np.newaxis]
                        if modality == 'T1':
                            img = np.concatenate((img, AimgT1), axis = 3)
                        elif modality == 'T2':
                            img = np.concatenate((img, AimgT2), axis = 3)
                        elif modality == 'both':
                            img = np.concatenate((img, AimgT1, AimgT2), axis = 3)
                        Aseg = Aseg[..., np.newaxis]
                        seg = np.concatenate((seg, Aseg), axis = 3)
                
                # sample patches and perform pre autmentation
                img = np.rollaxis(img, 3, 0)
                seg = np.rollaxis(seg, 3, 0)
                coordmaps = np.rollaxis(coordmaps, 3, 0)
                idside = subjid + '_' + side
                sample_phase = 'test'
                sample = {'id': idside, 'image': img, 'seg': seg, 'coordmaps': coordmaps, 'affine': affine, 'type': sample_phase}
                test_patches = p.GeneratePatches(sample, reorganize_config)
                test_batch_size = loaders_config['test_batch_size']
                testloader = DataLoader(test_patches, batch_size = test_batch_size, shuffle = False, num_workers = loaders_config['num_workers'])

                image_shape = sample['image'].shape[1:]
                im_shape_pad = image_shape + half_patch * 2
                prob = np.zeros([num_class] + list(im_shape_pad))
                prob_T1 = np.zeros([num_class] + list(im_shape_pad))
                prob_T2 = np.zeros([num_class] + list(im_shape_pad))
                rep = np.zeros([num_class] + list(im_shape_pad))
                pred_list = []

                for z, patch_batched in enumerate(testloader):

                    print("    Batch " + str(z) + " of " + str(len(testloader)) + " batches (test batch size: " + str(test_batch_size) + ").", end="\r")
                    img = patch_batched['image']
                    seg = patch_batched['seg']
                    coordmaps = patch_batched['coordmaps']
                    cpts = patch_batched['cpt']

                    # pad image if needed
                    if preserve_size:
                        if model_padding == 0:
                            pad_size = 0
                            for ii in range(model_num_levels):
                                pad_size = pad_size + 4*2**ii
                            pad_size = pad_size - 2*2**ii
                            img = F.pad(input=img, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
                            coordmaps = F.pad(input=coordmaps, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
        
        
                    # perform perdiction based on the training method.
                    if with_atlas:
        
                        # reorganize the patches
                        nbt, npt, nx, ny, nz = seg.size()
                        natlas = seg.shape[1] - 1
                        nmodality = int(img.shape[1]/natlas)
                        seg_targets = seg[:, 0, ...]
                        pat_targets = img[:, 0:nmodality, ...]
                        seg_atlases = seg[:, 1:npt, ...]
                        pat_atlases = img[:, nmodality:(nmodality*npt), ...]

                        # move data to cuda if available
                        seg_atlases = seg_atlases.to(device)
                        seg_targets = seg_targets.to(device)
                        pat_atlases = pat_atlases.to(device)
                        pat_targets = pat_targets.to(device)
                        coordmaps = coordmaps.to(device)

                        # generate model input
                        model_input = None
                        for i in range(nbt):
                            for j in range(npt-1):
                                pat_atlas = pat_atlases[i:i+1, nmodality*j:nmodality*(j+1), ...]
                                pat_target = pat_targets[i:i+1, ...]
                                pat = torch.cat((pat_target, pat_atlas), dim = 1)
                                if model_input is None:
                                    model_input = pat
                                else:
                                    model_input = torch.cat((model_input, pat), dim = 0)

                        # forward pass with both modalities
                        output, _, _ = model(model_input, seg_atlases, coordmaps, device)
                        if deep_supervision:
                            output = output[-1]
                            
                        inter_sigmoid = model_config.get('inter_sigmoid', False)
                        if model_config['fine_tunning_layers']:
                            if model_config['final_sigmoid']:
                                probability = output.cpu().numpy()
                            else:
                                probability = F.softmax(output, dim = 1).cpu().numpy()
                        else:
                            if inter_sigmoid:
                                probability = output.cpu().numpy()
                            else:
                                probability = F.softmax(output, dim = 1).cpu().numpy()

                        if modality == 'T1' or modality == 'T2':
                        
                            # No need to perform multitmodal testing
                            probability_T1 = probability
                            probability_T2 = probability
                            
                        else:
                               
                            # forward pass with both only T1
                            model_input_T1 = model_input * 1.0
                            model_input_T1[:, 1:2, :, :, :] = torch.randn(nbt*natlas, 1, nx, ny, nz, device=device)
                            model_input_T1[:, 3:4, :, :, :] = torch.randn(nbt*natlas, 1, nx, ny, nz, device=device)
                            output, _, _ = model(model_input_T1, seg_atlases, coordmaps, device)
                            if deep_supervision:
                                output = output[-1]

                            if model_config['fine_tunning_layers']:
                                if model_config['final_sigmoid']:
                                    probability_T1 = output.cpu().numpy()
                                else:
                                    probability_T1 = F.softmax(output, dim = 1).cpu().numpy()
                            else:
                                if inter_sigmoid:
                                    probability_T1 = output.cpu().numpy()
                                else:
                                    probability_T1 = F.softmax(output, dim = 1).cpu().numpy()

                            # forward pass with both only T2
                            model_input_T2 = model_input * 1.0
                            model_input_T2[:, 0:1, :, :, :] = torch.randn(nbt*natlas, 1, nx, ny, nz, device=device)
                            model_input_T2[:, 2:3, :, :, :] = torch.randn(nbt*natlas, 1, nx, ny, nz, device=device)
                            output, _, _ = model(model_input_T2, seg_atlases, coordmaps, device)
                            if deep_supervision:
                                output = output[-1]

                            if model_config['fine_tunning_layers']:
                                if model_config['final_sigmoid']:
                                    probability_T2 = output.cpu().numpy()
                                else:
                                    probability_T2 = F.softmax(output, dim = 1).cpu().numpy()
                            else:
                                if inter_sigmoid:
                                    probability_T2 = output.cpu().numpy()
                                else:
                                    probability_T2 = F.softmax(output, dim = 1).cpu().numpy()

                                    
                            # delete some variables
                            del model_input_T1
                            del model_input_T2
                            
                        # delete some variables
                        del seg_atlases
                        del seg_targets
                        del pat_atlases
                        del pat_targets
                        del model_input
                        del coordmaps
                        #del weightmaps
                        del output
                        torch.cuda.empty_cache()
        
        
                    else: 
                
                        # forward pass with both modalities
                        input = torch.cat((img, coordmaps), dim = 1).type(torch.cuda.FloatTensor)
                        output = model(input)
                        if deep_supervision:
                            output = output[-1]

                        if model_config['final_sigmoid']:
                            probability = output.cpu().numpy()
                        else:
                            probability = F.softmax(output, dim = 1).cpu().numpy()

                        if modality == 'T1' or modality == 'T2':
                        
                            # No need to perform multitmodal testing
                            probability_T1 = probability
                            probability_T2 = probability
                            
                        else:
                            
                            # forward pass with only T1
                            img_T1 = img * 1.0
                            img_T1[:,1:2,:,:,:] = torch.randn(nbatch, 1, nx, ny, nz, device=device)
                            input = torch.cat((img_T1, coordmaps), dim = 1).type(torch.cuda.FloatTensor)
                            output = model(input)
                            if deep_supervision:
                                output = output[-1]

                            if model_config['final_sigmoid']:
                                probability_T1 = output.cpu().numpy()
                            else:
                                probability_T1 = F.softmax(output, dim = 1).cpu().numpy()

                            # forward pass with only T2
                            img_T2 = img * 1.0
                            img_T2[:,0:1,:,:,:] = torch.randn(nbatch, 1, nx, ny, nz, device=device)
                            input = torch.cat((img_T2, coordmaps), dim = 1).type(torch.cuda.FloatTensor)
                            output = model(input)
                            if deep_supervision:
                                output = output[-1]

                            if model_config['final_sigmoid']:
                                probability_T2 = output.cpu().numpy()
                            else:
                                probability_T2 = F.softmax(output, dim = 1).cpu().numpy()

                        
                    
                    #Crop the patch to only use the center part
                    patch_crop_size = reorganize_config['patch_crop_size']
                    if patch_crop_size > 0:
                        probability = probability[:,:,patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size]
                   
                        probability_T1 = probability_T1[:,:,patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size]
                        
                        probability_T2 = probability_T2[:,:,patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size]
                        
                    ## Assemble image in loop!
                    n, C, hp, wp, dp = probability.shape
                    half_shape = torch.tensor([hp, wp, dp])/2
                    hs, ws, ds = half_shape.type(torch.int)

                    for cpt, pred, pred_T1, pred_T2 in zip(list(cpts), list(probability), list(probability_T1), list(probability_T2)):
                        #if np.sum(pred)/hs/ws/ds < 0.1:
                        prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
                        prob_T1[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred_T1
                        prob_T2[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred_T2
                        rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1

                #Crop the image since we added padding when generating patches
                prob = prob[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                prob_T1 = prob_T1[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                prob_T2 = prob_T2[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                rep = rep[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                rep[rep==0] = 1e-6

                # Normalized by repetition
                prob = prob/rep
                seg_pred = np.argmax(prob, axis = 0).astype('float')
                prob_T1 = prob_T1/rep
                seg_pred_T1 = np.argmax(prob_T1, axis = 0).astype('float')
                prob_T2 = prob_T2/rep
                seg_pred_T2 = np.argmax(prob_T2, axis = 0).astype('float')

                # compute gdsc
                gdsc = computeGeneralizedDSC(Tseg, seg_pred)
                sys.stdout.write("\033[F") #back to previous line 
                sys.stdout.write("\033[K") #clear line 
                print("    Prediction accuracy both: ", gdsc)
                
                gdsc_T1 = computeGeneralizedDSC(Tseg, seg_pred_T1)
                print("    Prediction accuracy only T1: ", gdsc_T1)
                
                gdsc_T2 = computeGeneralizedDSC(Tseg, seg_pred_T2)
                print("    Prediction accuracy only T2: ", gdsc_T2)

                #nib.save(nib.Nifti1Image(prob, affine), osp.join(c.predictDir, phase + "_prob_" + str(subjid) + "_" + side + "_target.nii.gz"))
                nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(c.predictDir, phase + "_seg_" + str(subjid) + "_" + side + "_target_both.nii.gz" ))
                nib.save(nib.Nifti1Image(seg_pred_T1, affine), osp.join(c.predictDir, phase + "_seg_" + str(subjid) + "_" + side + "_target_onlyT1.nii.gz" ))
                nib.save(nib.Nifti1Image(seg_pred_T2, affine), osp.join(c.predictDir, phase + "_seg_" + str(subjid) + "_" + side + "_target_onlyT2.nii.gz" ))


print('Done')
                
                
                

