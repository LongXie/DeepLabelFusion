# -*- coding: utf-8 -*-
"""
Created on April 31 2021

@author: Long Xie
"""

import importlib
import torch
import torch.nn as nn



def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None, voxelwise_weight = None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    if voxelwise_weight is None:
        intersect = (input * target).sum(-1)
        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (input * input).sum(-1) + (target * target).sum(-1)
    
    else:
        voxelwise_weight = flatten(voxelwise_weight)
        intersect = ( voxelwise_weight * input * target).sum(-1)
        # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
        denominator = (voxelwise_weight * input * input).sum(-1) + (voxelwise_weight * target * target).sum(-1)
    
    
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    #denominator = (voxelwise_weight * input * input).sum(-1) + (voxelwise_weight * target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            #self.normalization = nn.Softmax(dim=1)
            self.normalization = None #nn.Softmax(dim=1)

    def dice(self, input, target, weight, voxelwise_weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target, voxelwise_weight = None):
        # get probabilities from logits
        if self.normalization:
            input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight, voxelwise_weight = voxelwise_weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797.
    For multi-class segmentation `weight` parameter can be used to assign different weights per class.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super().__init__(weight, sigmoid_normalization)

    def dice(self, input, target, weight, voxelwise_weight):
        if voxelwise_weight is not None:
            assert target.size() == voxelwise_weight.size()
            
        #target = get_one_hot_image(input, target)
        
        if voxelwise_weight is not None:
            voxelwise_weight = voxelwise_weight.unsqueeze(1)
            voxelwise_weight = voxelwise_weight.expand_as(input)
        return compute_per_channel_dice(input, target, weight=self.weight, voxelwise_weight = voxelwise_weight)


class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        
    def dice(self, input, target, weight, voxelwise_weight):
        
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        #w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l = 1 / (w_l).clamp(min=self.epsilon)
        #w_l = 1 / torch.sqrt(w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l
        
        if self.weight is not None:
            intersect = weight * intersect

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)
        
        return 2 * (intersect / denominator)

class GeneralizedDiceLossWeighted(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, sigmoid_normalization=True, epsilon=1e-6):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        
    def dice(self, input, target, weight, voxelwise_weight):
        
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        #w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l = 1 / (w_l).clamp(min=self.epsilon)
        #w_l = 1 / torch.sqrt(w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l
        
        if self.weight is not None:
            intersect = weight * intersect

        denominator = (input + target).sum(-1)
        denominator = weight * (denominator * w_l).clamp(min=self.epsilon)
        
        return 2 * (intersect / denominator)

    

def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    
    def _loss_class(class_name, module_name):
        m = importlib.import_module(module_name)
        clazz = getattr(m, class_name)
        return clazz
    
    assert 'loss' in config, 'Could not find loss function configuration'
    loss_config = config['loss']
    name = loss_config['name']
    
    ignore_index = loss_config.get('ignore_index', None)
    weight = loss_config.get('weight', None)
    
    if name == 'MONAI_GeneralizedDiceLoss':
        loss_class = _loss_class('GeneralizedDiceLoss', 'monai.losses')
        del loss_config['name']
        loss = loss_class(**loss_config)
        loss_config['name'] = name
        return loss
    elif name == 'GeneralizedDiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return GeneralizedDiceLoss(sigmoid_normalization=sigmoid_normalization)
    elif name == 'GeneralizedDiceLossWeighted':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        return GeneralizedDiceLoss(sigmoid_normalization=sigmoid_normalization)
    else:
        raise RuntimeError(f"Unsupported loss type: '{name}'.")
        
    