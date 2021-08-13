# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:20:17 2020

@author: Sadhana
"""
import torch
from torch import nn as nn
from torch.nn import functional as F


def conv3d(input_channels, output_channels, kernel_size, bias = True, padding=1, stride = 1):
    return nn.Conv3d(input_channels, output_channels, kernel_size, padding = padding, stride = stride, bias = bias)
    
    
def upconv3d( input_channels, output_channels, mode):
    
    if mode == 'transpose':
        return nn.ConvTranspose3d(input_channels, output_channels, kernel_size = 3, stride = (2,2,2), padding =1, output_padding = 1)
        
    else:
        return nn.Sequential(F.upsample(mode='trilinear', scale_factor=2, align_corners = False), nn.Conv3d(input_channels, output_channels, kernel_size = 1))
        
def normalization(input_channels,norm_type = 'gn'):
    
    if norm_type == 'bn':
        m = nn.BatchNorm3d(input_channels)
    elif norm_type == 'gn':
        m = nn.GroupNorm(1,input_channels)   
    return m
    
                  
class DownConv(nn.Module):

#Encoder building block that performs 2 convolutions and 1 max pool
#ReLU activation follows each convolution 

    def __init__(self, input_channels, output_channels, pooling=True, norm = 'gn', dropout_rate = 0.3, padding = 1):
        super(DownConv, self).__init__()

        self.conv1 = conv3d(input_channels, output_channels, kernel_size = 3, padding = padding)
        self.conv2 = conv3d(output_channels, output_channels, kernel_size = 3, padding = padding) 
        if norm == 'None':
            self.norm = None
        else:
            self.norm = normalization(output_channels, norm)        
        self.pooling = pooling        
        self.dropout = nn.Dropout3d(p = dropout_rate)
        
        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        
        if self.norm is not None:
            x = F.relu(self.norm(self.dropout(self.conv1(x))))
            x = F.relu(self.norm(self.dropout(self.conv2(x))))
        else:
            x = F.relu(self.dropout(self.conv1(x)))
            x = F.relu(self.dropout(self.conv2(x)))

        before_pool = x
        
        if self.pooling:
            x = self.pool(x)
       
        return x, before_pool
 
 
class UpConv(nn.Module):     
#A helper Module that performs 2 convolutions and 1 UpConvolution.
#A ReLU activation follows each convolution.

    def __init__(self,input_channels, output_channels,  norm, up_mode='transpose', dropout_rate = 0.1, padding = 1):
        
        super(UpConv, self).__init__()
 
        self.upconv = upconv3d(input_channels, output_channels, mode=up_mode)

        ## concatenation makes the input double again
        self.conv1 = conv3d(input_channels,output_channels, kernel_size = 3, padding = padding)
        self.conv2 = conv3d(output_channels, output_channels, kernel_size = 3, padding = padding)
        self.dropout = nn.Dropout3d(p = dropout_rate)
        if norm == 'None':
            self.norm = None
        else:
            self.norm = normalization(output_channels, norm)

    def center_crop(self, fea, target_size):
        _, _, fea_x, fea_y, fea_z = fea.size()
        diff_x = (fea_x - target_size[0]) // 2
        diff_y = (fea_y - target_size[1]) // 2
        diff_z = (fea_z - target_size[2]) // 2
        return fea[
            :, :, diff_x : (diff_x + target_size[0]), diff_y : (diff_y + target_size[1]), diff_z : (diff_z + target_size[2])
        ]
            
    def forward(self, x, from_encoder):
        
        #Up-sample
        x = self.upconv(x)
        #Crop features from encoder
        from_encoder = self.center_crop(from_encoder, x.shape[2:])
        #Concatenate
        x = torch.cat([x, from_encoder], 1)
        # Double convolution
        if self.norm is not None:
            x = F.relu(self.norm(self.dropout(self.conv1(x))))
            x = F.relu(self.norm(self.dropout(self.conv2(x))))
        else:
            x = F.relu(self.dropout(self.conv1(x)))
            x = F.relu(self.dropout(self.conv2(x)))
        return x
    
    
    
class sandwich(nn.Module):
    
    def __init__(self, input_channels, output_channels, norm = 'bn', kernel_size = 3, padding = 1):
        
        super(sandwich, self).__init__()
        
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size = kernel_size, padding = padding)
        if norm == 'None':
            self.norm = None
        else:
            self.norm = normalization(output_channels, norm)
            
    def forward(self, x):
        
        x = self.norm(F.relu(self.conv(x)))
        
        return x
    
    
        
class resblock(nn.Module):
    
    def __init__(self, input_channels, output_channels, norm, doTransform_on_short = True):
        
        super(resblock, self).__init__()
        
        self.sandwich1 = sandwich(input_channels, output_channels, norm, kernel_size = 3, padding = 1)
        self.sandwich2 = sandwich(output_channels, output_channels, norm, kernel_size = 3, padding = 1)
        self.doTransform_on_short = doTransform_on_short
        
        if doTransform_on_short:
            self.shortsandwich = sandwich(input_channels, output_channels, norm, kernel_size = 1, padding = 0)
            
                
    def forward(self, x):
            
        x_short = x
        if self.doTransform_on_short:
            x_short = self.shortsandwich(x_short)
        
        x = self.sandwich1(x)
        x = self.sandwich2(x)
            
        x = x + x_short
                
        return x
        
        
class CongDownConv(nn.Module):

#Encoder building block that performs 2 convolutions and 1 max pool
#ReLU activation follows each convolution 

    def __init__(self, input_channels, output_channels, pooling=True, norm = 'gn', dropout_rate = 0.3):
        super(CongDownConv, self).__init__()

        self.resblock = resblock(input_channels, output_channels, norm)       
        self.pooling = pooling
        
        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = self.resblock(x)
        before_pool = x
        
        if self.pooling:
            x = self.pool(x)
       
        return x, before_pool
        
class CongUpConv(nn.Module):     
#A helper Module that performs 2 convolutions and 1 UpConvolution.
#A ReLU activation follows each convolution.

    def __init__(self,input_channels, output_channels,  norm, up_mode='transpose', dropout_rate = 0.1):
        
        super(CongUpConv, self).__init__()
 
        self.upconv = upconv3d(input_channels, output_channels, mode=up_mode)

        ## concatenation makes the input double again
        self.resblock = resblock(input_channels, output_channels, norm)

    def forward(self, x, from_encoder):
        
        #Up-sample
        x = self.upconv(x)        
        #Concatenate
        x = torch.cat([x, from_encoder], 1)
        # resblock
        x = self.resblock(x)
        
        return x
    
class CongFinalConv(nn.Module):     
#A helper Module that performs 2 convolutions and 1 UpConvolution.
#A ReLU activation follows each convolution.

    def __init__(self,input_channels, output_channels, inter_channels, norm):
        
        super(CongFinalConv, self).__init__()
 
        self.sandwich1 = sandwich(input_channels, inter_channels, norm, kernel_size = 1, padding = 0)
        self.sandwich2 = sandwich(inter_channels, inter_channels, norm, kernel_size = 1, padding = 0)
        self.finalconv = nn.Conv3d(inter_channels, output_channels, kernel_size = 1)

    def forward(self, x):
        
        x = self.sandwich1(x)
        x = self.sandwich2(x)
        x = self.finalconv(x)
        
        return x
        
            