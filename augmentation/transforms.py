# -*- coding: utf-8 -*-
"""
Created on April 31 2021

@author: Long Xie
"""

import importlib
import torch
from torchvision.transforms import Compose


def get_transformer(config, phase):
    if phase == 'val':
        phase = 'test'

    assert phase in config, f'Cannot find transformer config for phase: {phase}'
    phase_config = config[phase]
    base_config = {}
    return Transformer(phase_config, base_config)


class Transformer:
    def __init__(self, phase_config, base_config):
        self.phase_config = phase_config
        self.config_base = base_config
        #self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform('raw')

    def label_transform(self):
        return self._create_transform('label')

    def weight_transform(self):
        return self._create_transform('weight')

    @staticmethod
    def _transformer_class(class_name, module_name):
        m = importlib.import_module(module_name)
        clazz = getattr(m, class_name)
        return clazz
    
    #'augmentation.transforms'

    def _create_transform(self, name):
        assert name in self.phase_config, f'Could not find {name} transform'
        return Compose([
            self._create_augmentation(c) for c in self.phase_config[name]
        ])

    def _create_augmentation(self, c):
        config = dict(self.config_base)
        config.update(c)
        name = c['name']
        #c['random_state'] = np.random.RandomState(self.seed)
        
        if name == 'MONAI_ToTensor':
            aug_class = self._transformer_class('ToTensor', 'monai.transforms')
            del c['name']
            aug = aug_class(**c)
            c['name'] = name
            return aug