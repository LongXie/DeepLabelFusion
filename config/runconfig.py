# This scripts houses some usual config and constants used in the network
import os
import shutil
import numpy as np

class Config(object):
    def __init__(self):
        
        # ------------------ Parameters --------------------#
        ## Directories and training image generation
        # Root directory that house the image
        self.rootDir = "/home/lxie/MultiModalDL/"
        self.expName = "UNet"
        
        # Directory that contain the code
        self.codeDir = self.rootDir + '/' + self.expName + "/python_code/"
        
        # Current fold
        self.fold = 0
        
        # Directory of the expriment
        self.expDir = self.rootDir + '/' + self.expName + '/fold_' + '{0:01d}'.format(self.fold)

        # Directories that contrain the rawdata
        self.train_val_csv = self.expDir + "/split.csv"
        self.rawdataDir = self.rootDir + '/DeepLFdataset/fold_' + '{0:01d}'.format(self.fold)
        self.rawdata7TT2Dir = self.rootDir + '/DeepLF7TT2dataset/fold_' + '{0:01d}'.format(self.fold)
        
        # Temporary directory
        self.tmpdir = '/tmp/'
        
    def force_create(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)


class Config_Run(Config):
    def __init__(self):
        # Set base init
        super().__init__()
       
        # Experiment ID
        self.runID =  504

        # yaml file
        self.config_file = os.path.join(self.codeDir, 'config', 'train_config_run' + '{0:03d}'.format(self.runID) + '.yaml')
        self.test_config_file = os.path.join(self.codeDir, 'config', 'test_config_run' + '{0:03d}'.format(self.runID) + '.yaml')
        
        # Directories that contain the sampled patches
        self.patchesDir = self.expDir + '/dataset/run_' + '{0:03d}'.format(self.runID)
        self.patchesDataDir = self.patchesDir + '/data/'
        self.train_patch_csv = self.patchesDir + '/training_patch_list.csv'
        self.val_patch_csv = self.patchesDir + '/validation_patch_list.csv'
        
        # Directories that contain the model and tfrecord
        self.checkpointDir = self.expDir + "/model/run_" + '{0:03d}'.format(self.runID)
        
        # prediction directory
        self.predictDir = self.expDir + "/predict/run_" + '{0:03d}'.format(self.runID)
        self.predict7TT2Dir = self.expDir + "/predict7TT2/run_" + '{0:03d}'.format(self.runID)
               
        
        
        
        
        
