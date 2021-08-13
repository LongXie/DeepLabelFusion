import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.utils import get_one_hot_image

import importlib
from . import utils
from utils.gen_utils import get_logger
        
        
class UNet3DTrainerGetModel:
    """3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, 
                 logger=None, tensorboard_formatter=None,
                 best_eval_score=None,
                 skip_train_validation=False, 
                 use_weighted_loss = 0, 
                 deep_supervision=False, 
                 deep_supervision_weights=None, 
                 preserve_size=False, 
                 model_config=None, 
                 multimodal_augment=False, 
                 multimodal_augment_prob=0.5, 
                 **kwargs):
        
        if logger is None:
            self.logger = get_logger('UNet3DTrainerGetModel', level=logging.DEBUG)
        else:
            self.logger = logger

        
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.use_weighted_loss = use_weighted_loss
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights
        self.preserve_size = preserve_size
        self.model_config = model_config
        self.multimodal_augment = multimodal_augment
        self.multimodal_augment_prob = multimodal_augment_prob
        
        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        #assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation
        
        self.train_losses = utils.RunningAverage()
        self.train_eval_scores = utils.RunningAverage()
        
    def fit(self, model, optimizer, lr_scheduler, loss_criterion, 
            eval_criterion, device, loaders, checkpoint_dir, 
            logger = None, tensorboard_formatter = None):
        
        if logger is None:
            self.logger = get_logger('UNet3DMultimodalTrainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(f'eval_score_higher_is_better: {self.eval_score_higher_is_better}')
        
        # initialize
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_formatter = tensorboard_formatter
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'logs'))
        
        # reset losses accumulate sum
        self.train_losses.reset()
        self.train_eval_scores.reset()
        
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        
        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            
            #torch.cuda.empty_cache()
            input, target, coordmaps, weight = self._split_training_batch(t)

            # multimodal augmentation
            if self.multimodal_augment:
                nbatch, nc, nx, ny, nz = input.size()
                for i in range(nbatch):
                    if np.random.rand(1) < self.multimodal_augment_prob:
                        if np.random.rand(1) < 0.5:
                            rm_mod_idx = 0
                        else:
                            rm_mod_idx = 1
                        input[i,rm_mod_idx:(rm_mod_idx+1),:,:,:] = torch.randn(1, nx, ny, nz, device=self.device)
            
            output, loss = self._forward_pass(torch.cat((input, coordmaps), dim = 1), target, coordmaps, weight)

            self.train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    # crop target according to output size (for 0 padding)
                    #target = self._center_crop(target, output.shape[2:])
                    eval_score = self.eval_criterion(output, target)
                    self.train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {self.train_losses.avg}. Evaluation score: {self.train_eval_scores.avg}')
                self._log_stats('train', self.train_losses.avg, self.train_eval_scores.avg)
                #self._log_params()
                self._log_images(input, target, output)
                
                # reset running average
                self.train_losses.reset()
                self.train_eval_scores.reset()
            
            if self.num_iterations % self.validate_after_iters == 0:
                
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)
                
            # clean up GPU memory
            #del output, input, target, coordmaps, loss
            #torch.cuda.empty_cache() 

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        # adjust learning rate if necessary after each epoch
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()
            
        return False
    
    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                self.logger.info(f'Validation iteration {i}')

                input, target, coordmaps, weight = self._split_training_batch(t)
                
                output, loss = self._forward_pass(torch.cat((input, coordmaps), dim = 1), target, coordmaps, weight)
                
                val_losses.update(loss.item(), self._batch_size(input))

                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                # crop target according to output size (for 0 padding)
                #target = self._center_crop(target, output.shape[2:])
                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            del input, target, output, loss, coordmaps
            torch.cuda.empty_cache()
                    
            self._log_stats('val', val_losses.avg, val_scores.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg
    
    def _split_training_batch(self, t):
        
        input = t['image'].to(self.device)
        target = t['seg'].to(self.device)
        coordmaps = t['coordmaps'].to(self.device)
        weight = None
        
        if self.preserve_size:
            model_padding = self.model_config.get('padding', 1)
            if model_padding == 0:
                model_num_levels = self.model_config['num_levels']
                pad_size = 0
                for ii in range(model_num_levels):
                    pad_size = pad_size + 4*2**ii
                pad_size = pad_size - 2*2**ii
                input = F.pad(input=input, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
                coordmaps = F.pad(input=coordmaps, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
        
        
        return input, target, coordmaps, weight
    
    def _center_crop(self, input, target_size):
        _, input_x, input_y, input_z = input.size()
        diff_x = (input_x - target_size[0]) // 2
        diff_y = (input_y - target_size[1]) // 2
        diff_z = (input_z - target_size[2]) // 2
        return input[
            :, diff_x : (diff_x + target_size[0]), diff_y : (diff_y + target_size[1]), diff_z : (diff_z + target_size[2])
        ]
    
    def _forward_pass(self, input, target, coordmaps = None, weight=None):
        # forward pass
        output = self.model(input)
        #output = self.model(input, coordmaps)
        
        # crop target according to output size (for 0 padding)
        #target = self._center_crop(target, output.shape[2:])
        
        if self.deep_supervision:
            output_ds = output
            output = output[-1]
            
        # convert the target to one hot image
        target = get_one_hot_image(output, target)

        # compute the loss
        if self.deep_supervision is False:
            if weight is None:
                loss = self.loss_criterion(output, target)
            else:
                loss = self.loss_criterion(output, target, weight)
        else:
            if weight is None:
                loss = self.deep_supervision_weights[-1] * self.loss_criterion(output, target)
                for i in range(len(output_ds) - 1):
                    loss = loss + self.deep_supervision_weights[i] * self.loss_criterion(output_ds[i], target)
            else:
                loss = self.deep_supervision_weights[-1] * self.loss_criterion(output, targets, weight)
                for i in range(len(output_ds) - 1):
                    loss = loss + self.deep_supervision_weights[i] * self.loss_criterion(output_ds[i], target, weight)
            

        return output, loss
    
    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction):
        
        # for images
        inputs_map = {
            'inputs': input[0:10,:],
            'predictions': prediction[0:10,:]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')
                
                
        # for segmentations
        # get predicted segmentation
        probability = F.softmax(prediction, dim = 1)
        _, preds = torch.max(probability, 1)
        
        inputs_map = {
            'targetssegs': target[0:10,:],
            'predsegs': preds[0:10, :]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        color_transform = utils.Colorize(prediction.size(1))
                
        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch, normalize = False):
                imagetensor = torch.from_numpy(image)
                self.writer.add_image(tag, color_transform(imagetensor), self.num_iterations)
                

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)
    
    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
        
        
class UNet3DMultimodalTrainer:
    """Multimoal 3D UNet trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, max_num_epochs=100, max_num_iterations=1e5,
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 logger=None, tensorboard_formatter=None, skip_train_validation=False, 
                 use_weighted_loss = 0, 
                 deep_supervision=False, 
                 deep_supervision_weights=None, 
                 preserve_size=False, 
                 model_config=None, 
                 rm_mod_idx = 0, 
                 mod_alpha = 0.5, 
                 filltype = 'randn', 
                 mod_loss = 'self',
                 **kwargs):
        if logger is None:
            self.logger = get_logger('UNet3DMultimodalTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.use_weighted_loss = use_weighted_loss
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights
        self.preserve_size = preserve_size
        self.model_config = model_config
        
        # multimodal work
        self.rm_mod_idx = rm_mod_idx
        self.mod_alpha = mod_alpha
        self.filltype = filltype
        self.mod_loss = mod_loss
        
        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        #assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation
        
        self.train_losses = utils.RunningAverage()
        self.train_eval_scores = utils.RunningAverage()
        
    def fit(self, model, optimizer, lr_scheduler, loss_criterion, 
            eval_criterion, device, loaders, checkpoint_dir, 
            logger = None, tensorboard_formatter = None):
        
        if logger is None:
            self.logger = get_logger('UNet3DMultimodalTrainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(f'eval_score_higher_is_better: {self.eval_score_higher_is_better}')
        
        # initialize
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_formatter = tensorboard_formatter
        self.writer = SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, 'logs'))
        
        # reset losses accumulate sum
        self.train_losses.reset()
        self.train_eval_scores.reset()
        
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        
        # sets the model in training mode
        self.model.train()

        for i, t in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            
            #torch.cuda.empty_cache()
            input, target, coordmaps, weight = self._split_training_batch(t)

            output, outputmd, loss, inputmd = self._forward_pass(torch.cat((input, coordmaps), dim = 1), target, coordmaps, weight)

            self.train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    # crop target according to output size (for 0 padding)
                    #target = self._center_crop(target, output.shape[2:])
                    eval_score = self.eval_criterion(output, target)
                    self.train_eval_scores.update(eval_score.item(), self._batch_size(input))

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {self.train_losses.avg}. Evaluation score: {self.train_eval_scores.avg}')
                self._log_stats('train', self.train_losses.avg, self.train_eval_scores.avg)
                #self._log_params()
                self._log_images(input, target, output, outputmd, inputmd)
                
                # reset running average
                self.train_losses.reset()
                self.train_eval_scores.reset()
            
            if self.num_iterations % self.validate_after_iters == 0:
                
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)
                
            # clean up GPU memory
            #del output, input, target, coordmaps, loss
            #torch.cuda.empty_cache() 

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1

        # adjust learning rate if necessary after each epoch
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()
            
        return False
    
    def validate(self, val_loader):
        self.logger.info('Validating...')

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                self.logger.info(f'Validation iteration {i}')

                input, target, coordmaps, weight = self._split_training_batch(t)
                
                output, outputmd, loss, _ = self._forward_pass(torch.cat((input, coordmaps), dim = 1), target, coordmaps, weight)
                
                val_losses.update(loss.item(), self._batch_size(input))

                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                # crop target according to output size (for 0 padding)
                #target = self._center_crop(target, output.shape[2:])
                eval_score = self.eval_criterion(output, target)
                val_scores.update(eval_score.item(), self._batch_size(input))

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            del input, target, output, loss, coordmaps
            torch.cuda.empty_cache()
                    
            self._log_stats('val', val_losses.avg, val_scores.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
            return val_scores.avg
    
    def _split_training_batch(self, t):
        
        input = t['image'].to(self.device)
        target = t['seg'].to(self.device)
        coordmaps = t['coordmaps'].to(self.device)
        weight = None
        
        if self.preserve_size:
            model_padding = self.model_config.get('padding', 1)
            if model_padding == 0:
                model_num_levels = self.model_config['num_levels']
                pad_size = 0
                for ii in range(model_num_levels):
                    pad_size = pad_size + 4*2**ii
                pad_size = pad_size - 2*2**ii
                input = F.pad(input=input, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
                coordmaps = F.pad(input=coordmaps, pad=(pad_size,pad_size,pad_size,pad_size,pad_size,pad_size), mode='constant', value=0)
        
        
        return input, target, coordmaps, weight
    
    def _center_crop(self, input, target_size):
        _, input_x, input_y, input_z = input.size()
        diff_x = (input_x - target_size[0]) // 2
        diff_y = (input_y - target_size[1]) // 2
        diff_z = (input_z - target_size[2]) // 2
        return input[
            :, diff_x : (diff_x + target_size[0]), diff_y : (diff_y + target_size[1]), diff_z : (diff_z + target_size[2])
        ]
    
    def _forward_pass(self, input, target, coordmaps = None, weight=None):
        # forward pass
        output = self.model(input)
        #output = self.model(input, coordmaps)
        
        # crop target according to output size (for 0 padding)
        #target = self._center_crop(target, output.shape[2:])
        
        if self.deep_supervision:
            output_ds = output
            output = output[-1]
            
        
        # forward pass for the one without one of the modalities
        nbatch, nmod, nx, ny, nz = input.size()
        inputmd = input * 1.0
        if self.filltype.lower() == 'randn':
            if self.rm_mod_idx == -1:
                for i in range(nbatch):
                    if np.random.rand(1) > 0.5:
                        rm_mod_idx = 0
                    else:
                        rm_mod_idx = 1
                    inputmd[i,rm_mod_idx:(rm_mod_idx+1),:,:,:] = torch.randn(1, nx, ny, nz, device=self.device)
            else:
                inputmd[:,self.rm_mod_idx:(self.rm_mod_idx+1),:,:,:] = torch.randn(nbatch, 1, nx, ny, nz, device=self.device)
        elif self.filltype.lower() == 'zero':
            if self.rm_mod_idx == -1:
                for i in range(nbatch):
                    if np.random.rand(1) > 0.5:
                        rm_mod_idx = 0
                    else:
                        rm_mod_idx = 1
                    inputmd[i,rm_mod_idx:(rm_mod_idx+1),:,:,:] = 0
            else:
                inputmd[:,self.rm_mod_idx:(self.rm_mod_idx+1),:,:,:] = 0
        elif self.filltype.lower() == 'dupl':
            if self.rm_mod_idx == 0:
                inputmd[:,0:1,:,:,:] = inputmd[:,1:2,:,:,:]
            elif self.rm_mod_idx == 1:
                inputmd[:,1:2,:,:,:] = inputmd[:,0:1,:,:,:]
            elif self.rm_mod_idx == -1:
                for i in range(nbatch):
                    if np.random.rand(1) > 0.5:
                        inputmd[i,0:1,:,:,:] = inputmd[i,1:2,:,:,:]
                    else:
                        inputmd[i,1:2,:,:,:] = inputmd[i,0:1,:,:,:]
        outputmd = self.model(inputmd)
        
        #elif self.filltype.lower() == 'repzero':
        #    mulvector = torch.ones(2, device=self.device)
        #    mulvector[self.rm_mod_idx] = 0
        #    mulvector = mulvector[None, :, None, None, None]
        #    outputmd = self.model(input * mulvector)
        #    inputmd = input * mulvector
        
        if self.deep_supervision:
            outputmd_ds = outputmd
            outputmd = outputmd[-1]
            
        # convert the target to one hot image
        target = get_one_hot_image(output, target)

        # compute the loss
        mod_alpha = self.mod_alpha
        if self.mod_loss == 'MSELoss':
            mod_loss = MSELoss()
            
        if self.deep_supervision is False:
            if weight is None:
                if self.mod_loss == 'MSELoss':
                    loss = self.loss_criterion(output, target) + mod_alpha * mod_loss(outputmd, output)
                else:
                    loss = self.loss_criterion(output, target) + mod_alpha * self.loss_criterion(outputmd, target)
            else:
                if self.mod_loss == 'MSELoss':
                    loss = self.loss_criterion(output, target, weight) + mod_alpha * mod_loss(outputmd, output, weight)
                else:
                    loss = self.loss_criterion(output, target, weight) + mod_alpha * self.loss_criterion(outputmd, target, weight)
        else:
            if weight is None:
                if self.mod_loss == 'MSELoss':
                    loss = self.deep_supervision_weights[-1] * (self.loss_criterion(output, target) + 
                                                                mod_alpha * mod_loss(outputmd, output))
                    #print(mod_alpha * mod_loss(outputmd, output))
                    
                    for i in range(len(output_ds) - 1):
                        loss = loss + self.deep_supervision_weights[i] * (self.loss_criterion(output_ds[i], target) + 
                                                                          mod_alpha * mod_loss(outputmd_ds[i], output_ds[i])) 
                        #print(mod_alpha * mod_loss(outputmd_ds[i], output_ds[i]))
                else:
                    loss = self.deep_supervision_weights[-1] * (self.loss_criterion(output, target) + 
                                                                mod_alpha * self.loss_criterion(outputmd, target))
                    for i in range(len(output_ds) - 1):
                        loss = loss + self.deep_supervision_weights[i] * (self.loss_criterion(output_ds[i], target) + 
                                                                          mod_alpha * self.loss_criterion(outputmd_ds[i], target))
            else:
                if self.mod_loss == 'MSELoss':
                    loss = self.deep_supervision_weights[-1] * (self.loss_criterion(output, target, weight) + 
                                                                mod_alpha * mod_loss(outputmd, output, weight))
                    for i in range(len(output_ds) - 1):
                        loss = loss + self.deep_supervision_weights[i] * (self.loss_criterion(output_ds[i], target, weight) + 
                                                                          mod_alpha * mod_loss(outputmd_ds[i], output_ds[i], weight))  
                else:
                    loss = self.deep_supervision_weights[-1] * (self.loss_criterion(output, seg_targets, weight) + 
                                                                mod_alpha * self.loss_criterion(outputmd, target, weight))
                    for i in range(len(output_ds) - 1):
                        loss = loss + self.deep_supervision_weights[i] * (self.loss_criterion(output_ds[i], target, weight) +
                                                          mod_alpha * self.loss_criterion(outputmd_ds[i], target, weight))
        
        

        return output, outputmd, loss, inputmd
    
    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, predictionmd, inputmd):
        
        # for images
        inputs_map = {
            'inputs': input[0:10,:],
            'inputmd': inputmd[0:10,:],
            'predictions': prediction[0:10,:],
            'predictionsmd': predictionmd[0:10,:]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')
                
                
        # for segmentations
        # get predicted segmentation
        probability = F.softmax(prediction, dim = 1)
        _, preds = torch.max(probability, 1)
        probabilitymd = F.softmax(predictionmd, dim = 1)
        _, predsmd = torch.max(probabilitymd, 1)
        
        inputs_map = {
            'targetssegs': target[0:10,:],
            'predsegs': preds[0:10, :],
            'predsegsmd': predsmd[0:10, :]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        color_transform = utils.Colorize(prediction.size(1))
                
        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch, normalize = False):
                imagetensor = torch.from_numpy(image)
                self.writer.add_image(tag, color_transform(imagetensor), self.num_iterations)
                

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)
    
    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)

        

        
class DeepLabelFusionTrainer:
    """3D UNet trainer for label fusion.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, 
                 max_num_epochs=100, 
                 max_num_iterations=1e5,
                 validate_after_iters=100, 
                 log_after_iters=100,
                 validate_iters=None, 
                 num_iterations=1, 
                 num_epoch=0,
                 eval_score_higher_is_better=True, 
                 best_eval_score=None,
                 logger=None, 
                 tensorboard_formatter=None, 
                 skip_train_validation=False, 
                 use_weighted_loss = 0, 
                 step_after_iters = 1, 
                 use_designed_loss = 0, 
                 deep_supervision = False, 
                 deep_supervision_weights = None,
                 multimodal_augment = False,
                 multimodal_augment_prob = 0.5,
                 **kwargs):
        if logger is None:
            self.logger = get_logger('DeepLabelFusionTrainer', level=logging.DEBUG)
        else:
            self.logger = logger

        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better
        self.use_weighted_loss = use_weighted_loss
        self.step_after_iters = step_after_iters
        self.use_designed_loss = use_designed_loss
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights
        self.multimodal_augment = multimodal_augment
        self.multimodal_augment_prob = multimodal_augment_prob

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')

        #assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation
        
        self.train_losses = utils.RunningAverage()
        self.train_eval_scores = utils.RunningAverage()
        self.train_init_eval_scores = utils.RunningAverage()
        
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, model, optimizer, logger, config, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, skip_train_validation=False, use_weighted_loss = 0, step_after_iters = 1):
        logger.info(f"Loading checkpoint '{checkpoint_path}'...")
        state = utils.load_checkpoint(checkpoint_path, model, config)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   logger = logger, 
                   tensorboard_formatter=tensorboard_formatter,
                   skip_train_validation=skip_train_validation,
                   use_weighted_loss = use_weighted_loss, 
                   step_after_iters = step_after_iters, 
                   use_designed_loss = use_designed_loss)
    
    
    def fit(self, model, optimizer, lr_scheduler, loss_criterion, 
            eval_criterion, device, loaders, checkpoint_dir, 
            logger = None, tensorboard_formatter = None):
        
        if logger is None:
            self.logger = get_logger('DeepLabelFusionTrainer', level=logging.DEBUG)
        else:
            self.logger = logger
        self.logger.info(f'eval_score_higher_is_better: {self.eval_score_higher_is_better}')
        
        # initialize
        self.logger.info(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_formatter = tensorboard_formatter
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))
        
        # reset losses accumulate sum
        self.train_losses.reset()
        self.train_eval_scores.reset()
        self.train_init_eval_scores.reset()
        
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train(self.loaders['train'])

            if should_terminate:
                break

            self.num_epoch += 1

    def train(self, train_loader):
        """Trains the model for 1 epoch.

        Args:
            train_loader (torch.utils.data.DataLoader): training data loader

        Returns:
            True if the training should be terminated immediately, False otherwise
        """
        
        #train_losses = utils.RunningAverage()
        #train_eval_scores = utils.RunningAverage()
        #self.train_eval_scores.reset()
        #self.train_init_eval_scores.reset()

        # sets the model in training mode
        self.model.train()
        mode = "train"
        
        # zero the gradient 
        self.optimizer.zero_grad()
        
        for i, t in enumerate(train_loader):
            self.logger.info(
                f'Training iteration {self.num_iterations}. Batch {i}. Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            
            #print('Start')
            #torch.cuda.empty_cache()
            input, target, coordmaps, weight = self._split_training_batch(t, move_to_device = False)
            
            #print('Performing forward pass')
            output, loss, seg_targets, weightmap, output_init = self._forward_pass(input, target, coordmaps, weight, mode = mode)

            #print('Update loss')
            self.train_losses.update(loss.item(), self._batch_size(input))
            
            # compute gradients and update parameters
            #print('Perform backward pass')
            #self.optimizer.zero_grad()
            #loss = loss / self.step_after_iters
            loss.backward()
            
            #print('Update weight')
            #self.optimizer.step()
            
            
            if self.num_iterations % self.step_after_iters == 0:
                # Do a step once every step_after_iters iterations
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                del loss

            if self.num_iterations % self.log_after_iters == 0:
                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric as well as images in tensorboard will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                # compute eval criterion
                if not self.skip_train_validation:
                    eval_score = self.eval_criterion(output, seg_targets)
                    self.train_eval_scores.update(eval_score.item(), self._batch_size(input))
                    init_eval_score = self.eval_criterion(output_init, seg_targets)
                    self.train_init_eval_scores.update(init_eval_score.item(), self._batch_size(input))

                # log stats, params and images
                self.logger.info(
                    f'Training stats. Loss: {self.train_losses.avg}. Evaluation score: {self.train_eval_scores.avg}')
                self._log_stats('train', self.train_losses.avg, self.train_eval_scores.avg, self.train_init_eval_scores.avg)
                #self._log_params()
                self._log_images(input, seg_targets, output, weightmap, output_init)
                
                # reset running average
                self.train_losses.reset()
                self.train_eval_scores.reset()
                self.train_init_eval_scores.reset()
            
            if self.num_iterations % self.validate_after_iters == 0:
                
                # clean up GPU memory
                del output, seg_targets, weightmap
                torch.cuda.empty_cache() 
                
                # evaluate on validation set
                eval_score = self.validate(self.loaders['val'])
                
                # log current learning rate in tensorboard
                self._log_lr()
                # remember best validation metric
                is_best = self._is_best_eval_score(eval_score)

                # save checkpoint
                self._save_checkpoint(is_best)
                
            else:
                
                # clean up some variables
                del output, seg_targets, weightmap
                torch.cuda.empty_cache() 

            if self.max_num_iterations < self.num_iterations:
                self.logger.info(
                    f'Maximum number of iterations {self.max_num_iterations} exceeded. Finishing training...')
                return True

            self.num_iterations += 1
            
            # adjust learning rate if necessary at the end of each epoch
            #if isinstance(self.scheduler, ReduceLROnPlateau):
            #    self.scheduler.step(eval_score)
            #else:
            #    self.scheduler.step()
            
        # adjust learning rate if necessary after each epoch
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(eval_score)
        else:
            self.scheduler.step()        
            

        return False
    
    def validate(self, val_loader):
        self.logger.info('Validating...')
        mode = "val"

        val_losses = utils.RunningAverage()
        val_scores = utils.RunningAverage()

        with torch.no_grad():
            for i, t in enumerate(val_loader):
                self.logger.info(f'Validation iteration {i}')

                input_val, target_val, coordmaps_val, weight_val = self._split_training_batch(t, move_to_device = False)

                output_val, loss_val, seg_targets_val, _, _ = self._forward_pass(input_val, target_val, coordmaps_val, weight_val, mode = mode)
                val_losses.update(loss_val.item(), self._batch_size(input_val))

                # if model contains final_activation layer for normalizing logits apply it, otherwise both
                # the evaluation metric will be incorrectly computed
                #if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                #    output = self.model.final_activation(output)

                eval_score = self.eval_criterion(output_val, seg_targets_val)
                val_scores.update(eval_score.item(), self._batch_size(input_val))
                
                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break
                    
                if (i+1)%50 == 0:
                    break
                
                
            self._log_stats('val', val_losses.avg, val_scores.avg, val_scores.avg)
            self.logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
        
        del input_val, target_val, weight_val, output_val, loss_val, seg_targets_val
        torch.cuda.empty_cache()
        
        return val_scores.avg
    
    def _move_to_device(self, input):
        if isinstance(input, tuple) or isinstance(input, list):
            return tuple([_move_to_device(x) for x in input])
        else:
            return input.to(self.device)
    
    def _split_training_batch(self, t, move_to_device = True):
        input = t['image']
        target = t['seg']
        coordmaps = t['coordmaps']
        weight = None
        
        if move_to_device:
            input = self._move_to_device(input)
            target = self._move_to_device(target)
            coordmaps = self._move_to_device(coordmaps)
            weight = self._move_to_device(weight)
        
        return input, target, coordmaps, weight
    
    def _forward_pass(self, input, target, coordmaps, weight=None, mode="train"):
        
        # reorganize the patches
        nbt, npt, nx, ny, nz = target.size()
        _, nimgs, _, _, _ = input.size()
        nmod = nimgs // npt
        
        seg_targets = target[:, 0, ...]
        pat_targets = input[:, 0:nmod, ...]
        seg_atlases = target[:, 1:npt, ...]
        pat_atlases = input[:, nmod:(nmod*npt), ...]
        
        # move data to cuda if available
        seg_atlases = self._move_to_device(seg_atlases)
        seg_targets = self._move_to_device(seg_targets)
        coordmaps = self._move_to_device(coordmaps)
        #pat_atlases = self._move_to_device(pat_atlases)
        #pat_targets = self._move_to_device(pat_targets)

        # generate model input
        model_input = None
        for i in range(nbt):
            
            # extract target patch
            pat_target = pat_targets[i:i+1, ...]
            
            # multimodal augmentation in training if needed
            if self.multimodal_augment and mode == "train":
                
                augbool = np.random.rand(1)
                augidx = np.random.rand(1)
                if augidx < 0.5:
                    rm_mod_idx = 0
                else:
                    rm_mod_idx = 1
                
                if augbool < self.multimodal_augment_prob:
                    pat_target[:, rm_mod_idx:(rm_mod_idx+1),:,:,:] = torch.randn(1, nx, ny, nz, device=self.device)
            
            for j in range(npt-1):

                pat_atlas = pat_atlases[i:i+1, nmod*j:nmod*(j+1), ...]
                
                # multimodal augmentation in training if needed
                if self.multimodal_augment and mode == "train":
                    if augbool < self.multimodal_augment_prob:
                        pat_atlas[:, rm_mod_idx:(rm_mod_idx+1),:,:,:] = torch.randn(1, nx, ny, nz, device=self.device)
                
                pat = torch.cat((pat_target, pat_atlas), dim = 1)
                if model_input is None:
                    model_input = pat
                else:
                    model_input = torch.cat((model_input, pat), dim = 0)
                    
        model_input = self._move_to_device(model_input)
        
        # forward pass
        #nclass = 13
        output, weightmaps, output_init = self.model(model_input, seg_atlases, coordmaps, self.device)
        if self.deep_supervision:
            output_ds = output
            output = output[-1]
            
        # convert the target to one hot image
        seg_targets_orig = seg_targets
        seg_targets = get_one_hot_image(output, seg_targets)
        
        # compute weight if specified
        if self.use_weighted_loss == 1:
            # compute weight based on number of atlases that are different from target
            weight = torch.zeros(seg_targets.size())
            for i in range(npt-1):
                errormap = seg_atlases[:, i, ...] != seg_targets
                weight += errormap.type(torch.FloatTensor)
            weight = weight/(npt-1)
            weight = self._move_to_device(weight)
        elif self.use_weighted_loss == 2:
            # compute weight based on number of atlases that are different from target
            weight = torch.zeros(seg_targets.size())
            for i in range(npt-1):
                errormap = seg_atlases[:, i, ...] != seg_targets
                weight += errormap.type(torch.FloatTensor)
            weight = weight >= 1
            weight = weight.type(torch.FloatTensor) * 5.0 + 1
            weight = self._move_to_device(weight)
        elif self.use_weighted_loss == 3:
            # compute weight based on number of atlases that are different from target
            weight1 = torch.zeros(seg_targets.size())
            for i in range(npt-1):
                errormap = seg_atlases[:, i, ...] != seg_targets
                weight1 += errormap.type(torch.FloatTensor)
            weight1 = weight1 >= 1
            weight1 = weight1.type(torch.FloatTensor)
            
            # compute the entropy of atlas vots
            beta = 1.0
            nclass = output.size(1)
            prob = torch.zeros(output.size())
            for i in range(nclass):
                labelprob = (seg_atlases == i)
                labelprob = labelprob.type(torch.FloatTensor)
                labelprob = labelprob.sum(1)/nclass
                prob[:,i,...] = labelprob
            entropy = -prob * torch.log(prob).clamp(min=-1e6)
            weight2 = entropy.sum(1)
            
            # combine the weights
            alpha = 2.0
            beta = 1.0
            weight = 1 + weight1 * alpha + weight2 * beta
            weight = self._move_to_device(weight)
        
        
        # construct data for sepcific losses
        if self.use_designed_loss == 2 or self.use_designed_loss == 3:
            nclass = output.size(1)
            design_weight_maps = torch.zeros(nbt * (npt-1), nclass, nx, ny, nz)
            design_weight_maps = self._move_to_device(design_weight_maps)
            weightmaps_out = torch.zeros(nbt * (npt-1), nclass, nx, ny, nz)
            weightmaps_out = self._move_to_device(weightmaps_out)
            for i in range(nbt):
                for l in range(nclass):
                    labelmap_gt = seg_targets[i,...] == l
                    for j in range(npt-1):
                        idx = i*(npt-1) + j
                        labelmap_atlas = seg_atlases[i,j,...] == l
                        design_map = torch.mul(labelmap_gt, labelmap_atlas)
                        #design_map_tmp = torch.mul(labelmap_gt == 0, labelmap_atlas)
                        #design_weight_maps[idx,l,...] = design_map.type(torch.cuda.FloatTensor) - design_map_tmp.type(torch.cuda.FloatTensor)
                        design_weight_maps[idx,l,...] = design_map.type(torch.cuda.FloatTensor)
                        weightmaps_out[idx,l,...] = torch.mul(weightmaps[idx,l,...], labelmap_atlas.type(torch.cuda.FloatTensor))
        else:
            weightmaps_out = weightmaps
                        
        
        # compute the loss
        if self.deep_supervision is False:
            if self.use_designed_loss == 1:
                theta = 0.5
                loss = self.loss_criterion(output, seg_targets) + theta * self.loss_criterion(output_init, seg_targets)
            elif self.use_designed_loss == 2:
                mse_loss_weight = 2
                theta = mse_loss_weight / (nbt * (npt-1) * nx * ny * nz)

                #print("Here are the losses")
                #print(self.loss_criterion(output, seg_targets, weight))
                #print(theta * torch.nn.functional.mse_loss(weightmaps_out, design_weight_maps, reduction = 'sum'))

                if weight is None:
                    loss = self.loss_criterion(output, seg_targets) + theta * torch.nn.functional.mse_loss(weightmaps_out, design_weight_maps, reduction = 'sum')
                else:
                    loss = self.loss_criterion(output, seg_targets, weight) + theta * torch.nn.functional.mse_loss(weightmaps_out, design_weight_maps, reduction = 'sum')

            elif self.use_designed_loss == 3:
                loss_weight = 2
                theta = loss_weight / (nbt * (npt-1) * nx * ny * nz)

                #print("Here are the losses")
                #print(self.loss_criterion(output, seg_targets, weight))
                #print(theta * torch.nn.functional.mse_loss(weightmaps_out, design_weight_maps, reduction = 'sum'))

                if weight is None:
                    loss = self.loss_criterion(output, seg_targets) + theta * torch.nn.functional.smooth_l1_loss(weightmaps_out, design_weight_maps, reduction = 'sum')
                else:
                    loss = self.loss_criterion(output, seg_targets, weight) + theta * torch.nn.functional.smooth_l1_loss(weightmaps_out, design_weight_maps, reduction = 'sum')

            else:
                if weight is None:
                    loss = self.loss_criterion(output, seg_targets)
                else:
                    loss = self.loss_criterion(output, seg_targets, weight)
                    
        else:
            if weight is None:
                loss = self.deep_supervision_weights[-1] * self.loss_criterion(output, seg_targets)
                for i in range(len(output_ds) - 1):
                    loss = loss + self.deep_supervision_weights[i] * self.loss_criterion(output_ds[i], seg_targets)
            else:
                loss = self.deep_supervision_weights[-1] * self.loss_criterion(output, seg_targets, weight)
                for i in range(len(output_ds) - 1):
                    loss = loss + self.deep_supervision_weights[i] * self.loss_criterion(output_ds[i], seg_targets, weight)
            
            
        return output, loss, seg_targets_orig, weightmaps_out, output_init
    
    
    
    
    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg, init_eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg,
            f'{phase}_init_eval_score_avg': init_eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        self.logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, weightmap, prediction_init):
        
        # for images
        inputs_map = {
            'inputs': input[0:10,:],
            'weightmaps': weightmap[0:10, :],
            'predictions': prediction[0:10,:]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(tag, image, self.num_iterations, dataformats='CHW')
                
                
        # for segmentations
        # get predicted segmentation
        probability = F.softmax(prediction, dim = 1)
        _, preds = torch.max(probability, 1)
        probability_init = F.softmax(prediction_init, dim = 1)
        _, preds_init = torch.max(probability_init, 1)
        
        
        inputs_map = {
            'targetssegs': target[0:10,:],
            'predsegs': preds[0:10, :],
            'predsegs_init': preds_init[0:10, :]
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        color_transform = utils.Colorize(prediction.size(1))
                
        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch, normalize = False):
                imagetensor = torch.from_numpy(image)
                self.writer.add_image(tag, color_transform(imagetensor), self.num_iterations)
                

    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            self.logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best):
        utils.save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=self.logger)
    
    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)
        
        
        
        
def get_trainer(config):
    """
    Returns the model based on provided configuration
    :param config: (dict) a top level configuration object containing the 'model' key
    :return: an instance of the model
    """
    
    def _model_class(class_name, module_name):
        m = importlib.import_module(module_name)
        clazz = getattr(m, class_name)
        return clazz
    
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']
    name = trainer_config['name']
    
    if name == "UNet3DTrainer" or name == 'UNet3DMultimodalTrainer' or name == 'UNet3DTrainerGetModel' or name == "UNet3DLabelFusionTrainer" or name == "DeepLabelFusionTrainer":
        trainer_class = _model_class(name, 'model.trainer')
        return trainer_class(**trainer_config)
    else:
        raise RuntimeError(f"Unsupported model type: '{name}'.")
