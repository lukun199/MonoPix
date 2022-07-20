#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import sys
import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float, io
from .noise_data_tools import random_augmentation
from .aux_noise_utils import BaseDataSetH5, BaseDataSetFolder
from PIL import Image
import torchvision.transforms as transforms


def get_transform(Vary):
    transform_list = []

    #transform_list.append(transforms.RandomCrop(fineSize))
    if Vary == 1 and random.random() < 0.5:
        transform_list.append(transforms.ColorJitter(brightness=(0.5, 1.5)))
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

# Benchmardk Datasets: and SIDD
class BenchmarkTrain(BaseDataSetH5):
    # def initialize(self, h5_file, length, pch_size=128, args=None

    def __init__(self, opt):
        super(BenchmarkTrain, self).__init__(opt.h5file_path, None)

    def initialize(self, opt): # (opt.h5file_path, None, opt.pch_size, opt)
        self.pch_size = opt.patchSize
        self.opt = opt
        self.transform = get_transform(self.opt.vary)

    def __getitem__(self, index):
        num_images = self.num_images
        ind_img_gt = random.randint(0, num_images-1)
        ind_img_noisy = random.randint(0, num_images - 1)
        
        with h5.File(self.h5_path, 'r') as h5_file:
            im_gt = h5_file['clean_'+str(ind_img_gt)]  # dataset mismatch
            im_noisy = h5_file['noisy_'+str(ind_img_noisy)]
            im_gt, im_noisy = self.crop_patch(np.concatenate((im_gt, im_noisy), axis=-1))  # dataset mismatch
            
        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)  # augmentation not shared
        
        im_gt = Image.fromarray(im_gt)
        im_noisy = Image.fromarray(im_noisy)
        
        im_gt = self.transform(im_gt)
        im_noisy = self.transform(im_noisy)


        return {'A': im_gt, 'B': im_noisy, 'enhance_level_A2B': torch.rand(1, 1, 1),
                'enhance_level_B2A': torch.rand(1, 1, 1)}

    def name(self):
        return 'Noise_SIDD_Dataset'

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

class FakeTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128):
        super(FakeTrain, self).__init__(path_list, pch_size, length)

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        im_gt = img_as_float(cv2.imread(self.path_list[ind_im], 1)[:, :, ::-1])
        im_gt = self.crop_patch(im_gt)

        # data augmentation
        im_gt = random_augmentation(im_gt)[0]

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))

        return im_gt, im_gt, torch.zeros((1,1,1), dtype=torch.float32), torch.rand(im_noisy.shape[0], 1, 1) if self.args['IsControl']>0 else None

class PolyuTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128, mask=False):
        super(PolyuTrain, self).__init__(path_list, pch_size, length)
        self.mask = mask

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        path_noisy = self.path_list[ind_im]
        head, tail = os.path.split(path_noisy)
        path_gt = os.path.join(head, tail.replace('real', 'mean'))
        im_noisy = img_as_float(cv2.imread(path_noisy, 1)[:, :, ::-1])
        im_gt = img_as_float(cv2.imread(path_gt, 1)[:, :, ::-1])
        im_noisy, im_gt = self.crop_patch(im_noisy, im_gt)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.mask:
            return im_noisy, im_gt, torch.ones((1,1,1), dtype=torch.float32)
        else:
            return im_noisy, im_gt

    def crop_patch(self, im_noisy, im_gt):
        pch_size = self.pch_size
        H, W, _ = im_noisy.shape
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch_noisy = im_noisy[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        im_pch_gt = im_gt[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch_noisy, im_pch_gt
