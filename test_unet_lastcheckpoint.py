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

DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEFAULT_DEVICE = "cpu"
device = config.get('device', DEFAULT_DEVICE)
config['device'] = torch.device(device)
torch.cuda.set_device(0)
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


# In[19]:


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
                img = np.stack((TimgT1, TimgT2), axis = 3)
                seg = Tseg[..., np.newaxis]

                # sample patches and perform pre autmentation
                img = np.rollaxis(img, 3, 0)
                seg = np.rollaxis(seg, 3, 0)
                if modality == 'T1':
                    img = img[0:1, ...]
                elif modality == 'T2':
                    img = img[1:2, ...]
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
                rep = np.zeros([num_class] + list(im_shape_pad))
                pred_list = []

                for z, patch_batched in enumerate(testloader):

                    print("    Batch " + str(z) + " of " + str(len(testloader)) + " batches (test batch size: " + str(test_batch_size) + ").", end="\r")
                    img = patch_batched['image'].to(device)
                    seg = patch_batched['seg'].to(device)
                    coordmaps = patch_batched['coordmaps'].to(device)
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
        
                    # forward pass
                    input = torch.cat((img, coordmaps), dim = 1).type(torch.cuda.FloatTensor)
                    output = model(input)
                    if deep_supervision:
                        output = output[-1]
                    
                    if model_config['final_sigmoid']:
                        probability = output.cpu().numpy()
                    else:
                        probability = F.softmax(output, dim = 1).cpu().numpy()
                    
                    #Crop the patch to only use the center part
                    patch_crop_size = reorganize_config['patch_crop_size']
                    if patch_crop_size > 0:
                        probability = probability[:,:,patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size,
                                                  patch_crop_size:-patch_crop_size]

                    ## Assemble image in loop!
                    n, C, hp, wp, dp = probability.shape
                    half_shape = torch.tensor([hp, wp, dp])/2
                    hs, ws, ds = half_shape.type(torch.int)

                    for cpt, pred in zip(list(cpts), list(probability)):
                        #if np.sum(pred)/hs/ws/ds < 0.1:
                        prob[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += pred
                        rep[:,cpt[0] - hs:cpt[0] + hs, cpt[1] - ws:cpt[1] + ws, cpt[2] - ds:cpt[2] + ds] += 1

                #Crop the image since we added padding when generating patches
                prob = prob[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                rep = rep[:,half_patch[0]:-half_patch[0], half_patch[1]:-half_patch[1], half_patch[2]:-half_patch[2]]
                rep[rep==0] = 1e-6

                # Normalized by repetition
                prob = prob/rep
                seg_pred = np.argmax(prob, axis = 0).astype('float')
                prob = np.moveaxis(prob,0,-1)

                # compute gdsc
                gdsc = computeGeneralizedDSC(Tseg, seg_pred)
                sys.stdout.write("\033[F") #back to previous line 
                sys.stdout.write("\033[K") #clear line 
                print("    Prediction accuracy: ", gdsc)

                nib.save(nib.Nifti1Image(prob, affine), osp.join(c.predictDir, phase + "_prob_" + str(subjid) + "_" + side + "_target.nii.gz"))
                nib.save(nib.Nifti1Image(seg_pred, affine), osp.join(c.predictDir, phase + "_seg_" + str(subjid) + "_" + side + "_target.nii.gz" ))


print('Done')
                
                
                

