#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import numpy as np
import os
from functools import reduce
import operator
import matplotlib.pyplot as plt
import shutil
import csv
import glob
import nibabel as nib
import random
import utils.preprocess_data as p

# load configuration
import yaml
import config.runconfig as runconfig 
c = runconfig.Config_Run()

# Load and log experiment configuration
import torch
config = yaml.safe_load(open(c.config_file, 'r'))

# Set up GPU if available
DEFAULT_DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
device = config.get('device', DEFAULT_DEVICE)
config['device'] = torch.device(device)
print(config['device'])


# In[2]:


# reorganize data so that it can be read to used by data loader
reorganize_config = config['reorganize']
with_atlas = reorganize_config.get('with_atlas', False)

# generate folder saving information for each patch
c.force_create(c.patchesDir)
c.force_create(c.patchesDataDir)

# csv file
for phase in ['training', 'test']:
#for phase in [ 'test']:

    for subjdir in glob.glob(c.rawdataDir + "/" + phase + '/*'):

        for side in ['left', 'right']:

            # output current status
            subjid = subjdir.split('/')[-1]
            print("Reading " + phase + " " + side + " {}".format(subjid))

            # read T1 T2 and refseg images
            TargetT1 = os.path.join(c.rawdataDir, phase, subjid, 'mprage_to_tse_native_chunk_' + side + '_resampled.nii.gz')
            TimgT1hdr = nib.load(TargetT1)
            affine = TimgT1hdr.affine
            TimgT1 = TimgT1hdr.get_fdata()
            TargetT2 = os.path.join(c.rawdataDir, phase, subjid, 'tse_native_chunk_' + side + '_resampled.nii.gz')
            TimgT2 = nib.load(TargetT2).get_fdata()
            TargetSeg = os.path.join(c.rawdataDir, phase, subjid, 'refseg_' + side + '_chunk_resampled.nii.gz')
            Tseg = nib.load(TargetSeg).get_fdata()
            
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
            
            if with_atlas:
                for atlasdir in glob.glob(os.path.join(subjdir, 'multiatlas', 'tseg_' + side + '_train*')):

                    # output current status
                    atlasid = atlasdir.split('/')[-1]
                    idx = atlasid.split('train')[-1]
                    #print("    Sampling atlas:" + atlasid)

                    # read atlas T1 T2 and segmentation
                    AtlasT1 = os.path.join(atlasdir, 'atlas_to_native_mprage_resampled.nii.gz')
                    AimgT1 = nib.load(AtlasT1).get_fdata()
                    AtlasT2 = os.path.join(atlasdir, 'atlas_to_native_resampled.nii.gz')
                    AimgT2 = nib.load(AtlasT2).get_fdata()
                    AtlasSeg = os.path.join(atlasdir, 'atlas_to_native_segvote_resampled.nii.gz')
                    Aseg = nib.load(AtlasSeg).get_fdata()

                    # concatenate all images together
                    AimgT1 = AimgT1[..., np.newaxis]
                    AimgT2 = AimgT2[..., np.newaxis]
                    img = np.concatenate((img, AimgT1, AimgT2), axis = 3)
                    Aseg = Aseg[..., np.newaxis]
                    seg = np.concatenate((seg, Aseg), axis = 3)

            # sample patches and perform pre autmentation
            img = np.rollaxis(img, 3, 0)
            seg = np.rollaxis(seg, 3, 0)
            coordmaps = np.rollaxis(coordmaps, 3, 0)
            idside = subjid + '_' + side
            if phase == 'training':
                sample_phase = 'train'
            else:
                sample_phase = 'val'            
            sample = {'id': idside, 'image': img, 'seg': seg, 'coordmaps': coordmaps, 'affine': affine, 'type': sample_phase}
            print('Processing subject ' + str(sample['id']) + '(' + sample['type'] + ')')
            p.GeneratePatches(sample, reorganize_config)
            
            

