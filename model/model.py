# -*- coding: utf-8 -*-
"""
Created on April 30 2021

@author: Long Xie
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from model.unet_blocks import DownConv, UpConv,conv3d, CongDownConv, CongUpConv, CongFinalConv
import numpy as np
from torch.nn.parameter import Parameter

class UNet3DCoordmapsDeepSupervision4Level(nn.Module):

    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """

    def __init__(self,  in_channels, out_channels , init_feature_number = 32, num_levels = 4, norm = 'gn', down_dropout_rate = 0.3, up_dropout_rate = 0.1, final_sigmoid = False, **kwargs):
        super(UNet3DCoordmapsDeepSupervision4Level, self).__init__()

        encoder = []
        decoder = []
        output_channels = 0

        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):
            input_channels = in_channels if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, pooling = True, norm= norm, dropout_rate = down_dropout_rate)
            else:
                down_conv = DownConv(input_channels, output_channels, pooling = False, norm = norm, dropout_rate = down_dropout_rate)
            encoder.append(down_conv)

        self.encoders = nn.ModuleList(encoder)

        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            up_conv = UpConv(input_channels, output_channels, up_mode='transpose', norm = norm, dropout_rate = up_dropout_rate)
            decoder.append(up_conv)

        self.decoders = nn.ModuleList(decoder)

        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, out_channels, kernel_size = 1)
        
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None
            
        # deep supervision layers
        ds_uplayers0 = []
        n_channel = init_feature_number * 2 ** 3
        ds_uplayers0.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers0.append(nn.ConvTranspose3d(n_channel//2, n_channel//4, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers0.append(nn.ConvTranspose3d(n_channel//4, n_channel//8, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers0.append(nn.Conv3d(n_channel//8, out_channels, kernel_size = 1))
        self.ds_uplayers0 = nn.ModuleList(ds_uplayers0)
            
        ds_uplayers1 = []
        n_channel = init_feature_number * 2 ** 2
        ds_uplayers1.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers1.append(nn.ConvTranspose3d(n_channel//2, n_channel//4, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers1.append(nn.Conv3d(n_channel//4, out_channels, kernel_size = 1))
        self.ds_uplayers1 = nn.ModuleList(ds_uplayers1)
            
        ds_uplayers2 = []
        n_channel = init_feature_number * 2 ** 1
        ds_uplayers2.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
        ds_uplayers2.append(nn.Conv3d(n_channel//2, out_channels, kernel_size = 1))
        self.ds_uplayers2 = nn.ModuleList(ds_uplayers2)

        
    def forward(self, x):

        # concatanate coordmaps
        #coordmaps = coordmaps.type(torch.cuda.FloatTensor)
        #x = torch.cat((x, coordmaps), dim = 1)
        
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            x, before_pool = encoder(x)
            encoder_features.append(before_pool)

        # lowest resolution output
        outputs = []
        output0 = x
        for i, ds_uplayer in enumerate(self.ds_uplayers0):
            output0 = ds_uplayer(output0)
        outputs.append(output0)
            
        # decoder part
        for i, decoder in enumerate(self.decoders):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            before_pool = encoder_features[-(i+2)]
            x = decoder(x, before_pool)
            if i == 0:
                output1 = x
                for i, ds_uplayer in enumerate(self.ds_uplayers1):
                    output1 = ds_uplayer(output1)
                outputs.append(output1)
            elif i == 1:
                output2 = x
                for i, ds_uplayer in enumerate(self.ds_uplayers2):
                    output2 = ds_uplayer(output2)
                outputs.append(output2)

        x = self.final_conv(x)
        outputs.append(x)
        
        if self.final_activation is not None:
            for i, tmpoutput in enumerate(outputs):
                outputs[i] = self.final_activation(tmpoutput)
            
        return outputs
    

class LabelfusionUNet3DLabelWeightFineTunningUNetPerChannelSkipMaskCoordMapsBothDeepSupervision4LevelReLu(nn.Module):

    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_levels (int) : depth of the encoding part of the network
        norm ('bn' or 'gn'): Normalization type : Batch  or Group. default = 'gn'
    """

    def __init__(self,  in_channels, out_channels , nclass = 13, final_sigmoid = False, inter_sigmoid = False, init_feature_number = 32, num_levels = 4, norm = 'gn', rs = 0, fine_tunning_layers = True, down_dropout_rate = 0.3, up_dropout_rate = 0.1, tunning_dropout_rate = 0.0, num_tunning_levels = 3, init_tunning_feature_number = 32, **kwargs):
        super(LabelfusionUNet3DLabelWeightFineTunningUNetPerChannelSkipMaskCoordMapsBothDeepSupervision4LevelReLu, self).__init__()

        encoder = []
        decoder = []
        output_channels = 0
        self.in_channels = in_channels

        #Use 5 levels in the encoder path as suggested by the paper.
        # Last level doesn't have max-pooling
        # Create the encoder pathway
        for i in range(num_levels):
            input_channels = in_channels + 3 if i ==0 else output_channels
            output_channels = init_feature_number * 2 ** i
            if i < (num_levels - 1):
                down_conv = DownConv(input_channels, output_channels, pooling = True, norm= norm, dropout_rate = down_dropout_rate)
            else:
                down_conv = DownConv(input_channels, output_channels, pooling = False, norm = norm, dropout_rate = down_dropout_rate)
            encoder.append(down_conv)

        self.encoders = nn.ModuleList(encoder)

        # Create the decoder path. The length of the decove is equal to
        # num_levels - 1
        for i in range (num_levels - 1):
            input_channels = output_channels
            output_channels = input_channels // 2
            up_conv = UpConv(input_channels, output_channels, up_mode='transpose', norm = norm, dropout_rate = up_dropout_rate)
            decoder.append(up_conv)

        self.decoders = nn.ModuleList(decoder)

        # Final convolution layer to reduce the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(init_feature_number, out_channels, kernel_size = 1)
        
        if inter_sigmoid:
            self.inter_activation = nn.Sigmoid()
        else:
            self.inter_activation = None
        
            
        # construct a mean filter matrix
        if rs == 0:
            self.filter = None
        else:
            radius = abs(int(np.ceil(2 * rs + 1)))
            self.filter = torch.FloatTensor(1, 1, radius, radius, radius).fill_(1) /(radius*radius*radius)
            self.filter.requires_grad = False
         
        self.rs = rs
        
        # fine tunning layer
        self.nclass = nclass
        self.fine_tunning_layers = fine_tunning_layers
        if fine_tunning_layers:
            
            encoder = []
            decoder = []
            
            for i in range(num_tunning_levels):
                input_channels = nclass + 3 if i ==0 else output_channels
                output_channels = init_tunning_feature_number * 2 ** i
                if i < (num_tunning_levels - 1):
                    down_conv = DownConv(input_channels, output_channels, pooling = True, norm= norm, dropout_rate = down_dropout_rate)
                else:
                    down_conv = DownConv(input_channels, output_channels, pooling = False, norm = norm, dropout_rate = down_dropout_rate)
                encoder.append(down_conv)

            self.tunning_encoders = nn.ModuleList(encoder)

            # Create the decoder path. The length of the decove is equal to
            # num_levels - 1
            for i in range (num_tunning_levels - 1):
                input_channels = output_channels
                output_channels = input_channels // 2
                up_conv = UpConv(input_channels, output_channels, up_mode='transpose', norm = norm, dropout_rate = up_dropout_rate)
                decoder.append(up_conv)

            self.tunning_decoders = nn.ModuleList(decoder)

            # Final convolution layer to reduce the number of output
            # channels to the number of labels
            self.tunning_final_conv = nn.Conv3d(init_tunning_feature_number, nclass, kernel_size = 1)
            
            
            # deep supervision layers
            tunning_ds_uplayers0 = []
            n_channel = init_tunning_feature_number * 2 ** 3
            tunning_ds_uplayers0.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers0.append(nn.ReLU())
            tunning_ds_uplayers0.append(nn.ConvTranspose3d(n_channel//2, n_channel//4, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers0.append(nn.ReLU())
            tunning_ds_uplayers0.append(nn.ConvTranspose3d(n_channel//4, n_channel//8, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers0.append(nn.ReLU())
            tunning_ds_uplayers0.append(nn.Conv3d(n_channel//8, nclass, kernel_size = 1))
            self.tunning_ds_uplayers0 = nn.ModuleList(tunning_ds_uplayers0)
            
            tunning_ds_uplayers1 = []
            n_channel = init_tunning_feature_number * 2 ** 2
            tunning_ds_uplayers1.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers1.append(nn.ReLU())
            tunning_ds_uplayers1.append(nn.ConvTranspose3d(n_channel//2, n_channel//4, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers1.append(nn.ReLU())
            tunning_ds_uplayers1.append(nn.Conv3d(n_channel//4, nclass, kernel_size = 1))
            self.tunning_ds_uplayers1 = nn.ModuleList(tunning_ds_uplayers1)
            
            tunning_ds_uplayers2 = []
            n_channel = init_tunning_feature_number * 2 ** 1
            tunning_ds_uplayers2.append(nn.ConvTranspose3d(n_channel, n_channel//2, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1))
            tunning_ds_uplayers2.append(nn.ReLU())
            tunning_ds_uplayers2.append(nn.Conv3d(n_channel//2, nclass, kernel_size = 1))
            self.tunning_ds_uplayers2 = nn.ModuleList(tunning_ds_uplayers2)

        else:
            self.tunning_encoders = None
            self.tunning_decoders = None
            self.tunning_final_conv = None
            self.tunning_ds_uplayers0 = None
            self.tunning_ds_uplayers1 = None
            self.tunning_ds_uplayers2 = None
            
            
        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = None
            
        # smoothing filter for mask
        maskrs = 1
        maskradius = abs(int(np.ceil(2 * maskrs + 1)))
        self.maskfilter = torch.FloatTensor(1, 1, maskradius, maskradius, maskradius).fill_(1) /(maskradius*maskradius*maskradius)
        self.maskfilter.requires_grad = False
        self.maskrs = maskrs
            

    def forward(self, x, segs, coordmaps, device):

        # incorporate coordmaps to x
        nbatch, natlas, nx, ny, nz = segs.size()
        
        if device == "cpu":
            xinput = torch.zeros(nbatch*natlas, self.in_channels + 3, nx, ny, nz)
            f = self.filter
        else:
            xinput = torch.cuda.FloatTensor(nbatch*natlas, self.in_channels + 3, nx, ny, nz).fill_(0)
        coordmaps = coordmaps.type(torch.cuda.FloatTensor)
        for i in range(nbatch):
            for j in range(natlas):
                idx = i * natlas + j 
                xinput[idx, ...] = torch.cat((x[idx, ...], coordmaps[i, ...]), dim = 0)
        
        
        # Go through the UNet
        # Encoder part
        encoder_features = []
        for i,encoder in enumerate(self.encoders):
            xinput, before_pool = encoder(xinput)
            encoder_features.append(before_pool)

        # decoder part
        decoder_features = []
        for i, decoder in enumerate(self.decoders):
            # Indexing from the end of the array ( not level 5)
            # Pass the output from the corresponding encoder step
            before_pool = encoder_features[-(i+2)]
            xinput = decoder(xinput, before_pool)

        weightmaps = self.final_conv(xinput)
        
        
        if self.inter_activation is not None:
            weightmaps = self.inter_activation(weightmaps)
        
        
        # Multiply weights with the label maps
        # get some numbers
        #natlas = segs.size(1)
        #nbatch = segs.size(0)
        #nbatch, natlas, nx, ny, nz = segs.size()
        if device == "cpu":
            probmaps = torch.zeros(nbatch, natlas, self.nclass, nx, ny, nz)
            f = self.filter
        else:
            probmaps = torch.cuda.FloatTensor(nbatch, natlas, self.nclass, nx, ny, nz).fill_(0)
            f = self.filter
            if f is not None:
                f = f.to(device)
        for i in range(nbatch):
            for j in range(natlas):
                idx = i * natlas + j 
                for l in range(self.nclass):
                    
                    # generate label map
                    labelmap = segs[i, j, ...] == l
                    if device == "cpu":
                        labelmap = labelmap.type(torch.FloatTensor)
                    else:
                        labelmap = labelmap.type(torch.cuda.FloatTensor)
                    
                    # perform smoothing if needed
                    if f is not None:
                        labelmap = labelmap.unsqueeze(0).unsqueeze(0)
                        labelmap = F.conv3d(labelmap, f, bias=None, stride=1, padding=self.rs, dilation=1, groups=1)
                        labelmap = labelmap[0,...]
                    
                    # weight multiplication
                    if weightmaps.size(1) == self.nclass:
                        probmaps[i, j, l, ...] = weightmaps[idx, l, ...] * labelmap
                    else:
                        probmaps[i, j, l, ...] = weightmaps[idx, 0, ...] * labelmap
                    
        
        output = torch.mean(probmaps, dim = 1)
        
        # perform fine tunning
        if self.fine_tunning_layers:
            outputs = []
            output_init = output
            # get the gray scale images
            #xfinetune = x[np.arange(start=0, stop=natlas*nbatch, step=natlas), 0:2, ...]
            #coordmaps = coordmaps.type(torch.cuda.FloatTensor)
            output = torch.cat((output, coordmaps), dim = 1)
            encoder_features_tunning = []
            for i,encoder in enumerate(self.tunning_encoders):
                output, before_pool_tunning = encoder(output)
                encoder_features_tunning.append(before_pool_tunning)

            # lowest resolution output
            output0 = output
            for i, ds_uplayer in enumerate(self.tunning_ds_uplayers0):
                output0 = ds_uplayer(output0)
            output0 = output_init + output0
            outputs.append(output0)
                
            # decoder part
            for i, decoder in enumerate(self.tunning_decoders):
                # Indexing from the end of the array ( not level 5)
                # Pass the output from the corresponding encoder step
                before_pool_tunning = encoder_features_tunning[-(i+2)]
                output = decoder(output, before_pool_tunning)
                if i == 0:
                    output1 = output
                    for j, ds_uplayer in enumerate(self.tunning_ds_uplayers1):
                        output1 = ds_uplayer(output1)
                    output1 = output_init + output1
                    outputs.append(output1)
                elif i == 1:
                    output2 = output
                    for j, ds_uplayer in enumerate(self.tunning_ds_uplayers2):
                        output2 = ds_uplayer(output2)
                    output2 = output_init + output2
                    outputs.append(output2)

            # full resolution output
            output = self.tunning_final_conv(output)
            output = output_init + output
            outputs.append(output)
        
        
        if self.final_activation is not None:
            for i, tmpoutput in enumerate(outputs):
                outputs[i] = self.final_activation(tmpoutput)
            
        # generate mask and multiply
        outputs_final = []
        for i in range(len(outputs)):
            outputs_final.append(torch.cuda.FloatTensor(output.size()))
        maskf = self.maskfilter
        maskf = maskf.to(device)
        for i in range(nbatch):
            for l in range(self.nclass):
                mask = segs[i,...] == l
                mask = mask.sum(0)
                mask = mask>0
                mask = mask.type(torch.cuda.FloatTensor)
                #print(mask.sum())
                mask = mask.unsqueeze(0).unsqueeze(0)
                mask = F.conv3d(mask, maskf, bias=None, stride=1, padding=self.maskrs, dilation=1, groups=1)
                mask = mask[0,...] > 0.2
                mask = mask.type(torch.cuda.FloatTensor)
                #print(mask.sum())
                for k in range(len(outputs)):
                    outputs_final[k][i,l,... ] = outputs[k][i,l,... ] * mask
            
        return outputs_final, weightmaps, output_init
    
    
    

def get_model(config):
    """
    Returns the model based on provided configuration
    :param config: (dict) a top level configuration object containing the 'model' key
    :return: an instance of the model
    """
    
    def _model_class(class_name, module_name):
        m = importlib.import_module(module_name)
        clazz = getattr(m, class_name)
        return clazz
    
    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    name = model_config['name']
    
    if name == 'MONAI_UNET':
        model_class = _model_class('UNet', 'monai.networks.nets.unet')
        del model_config['name']
        model = model_class(**model_config)
        model_config['name'] = name
        return model
    elif name == "MONAI_DynUNet":
        model_class = _model_class('DynUNet', 'monai.networks.nets.unet')
        del model_config['name']
        model = model_class(**model_config)
        model_config['name'] = name
        return model
    elif name == "UNet3DCoordmapsDeepSupervision4Level" or name == "LabelfusionUNet3DLabelWeightFineTunningUNetPerChannelSkipMaskCoordMapsBothDeepSupervision4LevelReLu":
        model_class = _model_class(name, 'model.model')
        return model_class(**model_config)
    else:
        raise RuntimeError(f"Unsupported model type: '{name}'.")
        
    