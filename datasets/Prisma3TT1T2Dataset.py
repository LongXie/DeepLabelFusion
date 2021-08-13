import collections
import importlib
import pandas as pd
import csv
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import augmentation.transforms as transforms
from utils.gen_utils import get_logger

# load configuration
import yaml
import config.runconfig as runconfig 


logger = get_logger('Prisma3TT1T2Dataset')

class Prisma3TT1T2Dataset(Dataset):
    """
    Implementation of torch.utils.data.Dataset for the prisma 3T T1 T2 multimodal dataset
    
    """

    def __init__(self, csv_file, phase, config = None, modality = 'both'):
        """
        :param csv_path: path to the csv file containing raw data as well as labels
        :param phase: 'train' for training, 'val' for validation, 'test' for testing; data augmentation is performed
            only during the 'train' phase
        :param config: configuration
        :param modaliity: 'T1', 'T2' or 'both' to indicate which modality to use
        """
        # init
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.allinfo = pd.read_csv(csv_file, delimiter=',', header=None)
        self.modality = modality
        
        # load configurations
        transformer_config = config['loaders']['transformer']
        self.config = config['dataset']
        
        # get transformer
        if transformer_config:
            self.transformer = transforms.get_transformer(transformer_config, phase=phase)
            self.raw_transform = self.transformer.raw_transform()
        else:
            self.transformer = None
            self.raw_transform = None
        
        # apply label transform only for the training and validation sets
        if self.phase != 'test':
            if self.transformer:
                self.label_transform = self.transformer.label_transform()
            else:
                self.label_transform = None

    def __getitem__(self, idx):
        
        # get image id
        image_id = self.allinfo.iloc[idx,2]
        
        # read center point if available
        cpt = self.allinfo.iloc[idx, 3]
        
        # get phase id
        phase = self.allinfo.iloc[idx,4]
        
        # read target and atlas patches
        imageshdr = nib.load(self.allinfo.iloc[idx, 0])
        affine = imageshdr.affine
        images = imageshdr.get_data().astype('float32') 
        images = np.rollaxis(images, 3, 0)
        
        # choose modality
        if self.modality == 'T1':
            images = images[np.arange(0,images.shape[0],2), ...]
        elif self.modality == 'T2':
            images = images[np.arange(1,images.shape[0],2), ...]
        
        if self.raw_transform:
            images = self.raw_transform(images)
            
        # read coordinate maps
        coordmaps = nib.load(self.allinfo.iloc[idx, 2]).get_data().astype('float32') 
        coordmaps = np.rollaxis(coordmaps, 3, 0)
        if self.raw_transform:
            coordmaps = self.raw_transform(coordmaps)
        
        # read labels
        if self.phase != 'test':
            segs = nib.load(self.allinfo.iloc[idx, 1]).get_data()
            segs = np.rollaxis(segs, 3, 0)
            segs = np.squeeze(segs)
                       
            # transform the labels
            if self.label_transform:
                segs = self.label_transform(segs)
            
            sample = {'id': image_id, 'image':images, 'seg':segs, 'coordmaps':coordmaps, 'affine':affine, 'type':phase, 'cpt': cpt}
        else:
            sample = {'id': image_id, 'image':images, 'seg':[], 'coordmaps':coordmaps, 'affine':affine, 'type':phase, 'cpt': cpt}
        
        
        return sample
        
    def __len__(self):
        return len(self.allinfo)

    
def get_train_loaders(config, trainCSV, valCSV):
    """
    Returns dictionary containing the training and validation loaders
    (torch.utils.data.DataLoader) backed by the datasets.Prisma3T.Prisma3TDataset.

    :param 
        config: a top level configuration object containing the 'loaders' key
        trainCSV: CSV file that has the training data information
        valCSV: CSV file that has the validation data information
    :return: dict {
        'train': <train_loader>
        'val': <val_loader>
    }
    """
    assert 'loaders' in config, 'Could not find data loaders configuration'
    loaders_config = config['loaders']

    logger.info('Creating training and validation set loaders...')

    # number of workers
    num_workers = loaders_config.get('num_workers', 1)
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')
    
    # load batch size
    train_batch_size = loaders_config.get('train_batch_size', 1)
    val_batch_size = loaders_config.get('val_batch_size', 1)
    logger.info(f'Batch size for train loader: {train_batch_size}')
    logger.info(f'Batch size for validation loader: {val_batch_size}')
    
    # prepare training and validation datasets
    modality = loaders_config.get('modality', 'both')
    train_dataset = Prisma3TT1T2Dataset(
            trainCSV,
            phase='train',
            config=config,
            modality = modality)
    
    val_dataset = Prisma3TT1T2Dataset(
            valCSV,
            phase='val',
            config=config,
            modality = modality)
    
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train': DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=num_workers)
    }