import os.path
from numpy.lib import real
import torch
import random
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class BlendingDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, opt, is_for_train):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths, self.mask_paths, self.gt_paths, self.real_masks = [], [], [], []
        self.isTrain = is_for_train
        self._load_images_paths()
        self.transform = get_transform(opt)

    def _load_images_paths(self,):
        if self.isTrain == True:
            print('loading training file...')
            self.trainfile = os.path.join(self.opt.dataset_root, 'IHD_train.txt')
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace('.jpg','.png')
                    real_maskpath = mask_path.replace('masks','real_masks')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    self.image_paths.append(os.path.join(self.opt.dataset_root, line))
                    self.mask_paths.append(os.path.join(self.opt.dataset_root, mask_path))
                    self.gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))
                    self.real_masks.append(os.path.join(self.opt.dataset_root, real_maskpath))

        elif self.isTrain == False:
            print('loading test file...')
            self.trainfile = os.path.join(self.opt.dataset_root, 'IHD_test.txt')
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip()
                    name_parts = line.split('_')
                    mask_path = line.replace('composite_images', 'masks')
                    mask_path = mask_path.replace('.jpg','.png')
                    real_maskpath = mask_path.replace('masks','real_masks')
                    gt_path = line.replace('composite_images', 'real_images')
                    gt_path = gt_path.replace('_'+name_parts[-2]+'_'+name_parts[-1], '.jpg')
                    self.image_paths.append(os.path.join(self.opt.dataset_root, line))
                    self.mask_paths.append(os.path.join(self.opt.dataset_root, mask_path))
                    self.gt_paths.append(os.path.join(self.opt.dataset_root, gt_path))
                    self.real_masks.append(os.path.join(self.opt.dataset_root, real_maskpath))

    def __getitem__(self, index):
        comp = Image.open(self.image_paths[index]).convert('RGB')
        real = Image.open(self.gt_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('RGB')
        real_mask = Image.open(self.real_masks[index]).convert('RGB')

        comp = tf.resize(comp, [512, 512])
        mask = tf.resize(mask, [512, 512])
        real = tf.resize(real, [512, 512])
        real_mask = tf.resize(real_mask, [512, 512])

        #apply the same transform to composite and real images
        comp = self.transform(comp)
        mask = tf.to_tensor(mask)
        real = self.transform(real)
        real_mask = tf.to_tensor(real_mask)


        return {'comp': comp, 'mask': mask, 'real_mask':real_mask, 'real': real,'img_path':self.image_paths[index]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)