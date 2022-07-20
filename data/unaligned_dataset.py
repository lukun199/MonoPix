import torch
from torch import nn
import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset, store_dataset
import random
from PIL import Image
import numpy as np
import PIL
from pdb import set_trace as st

def pad_tensor(input):
    
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div  / 2)
            pad_bottom = int(height_div  - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

            padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
            input = padding(input).data
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.shape[2], input.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom

def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:,:, pad_top: height - pad_bottom, pad_left: width - pad_right]


class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + opt.source_dataset)
        self.dir_B = os.path.join(opt.dataroot, opt.phase + opt.target_dataset)

        # self.A_paths = make_dataset(self.dir_A)
        # self.B_paths = make_dataset(self.dir_B)
        self.A_imgs, self.A_paths = store_dataset(self.dir_A)
        self.B_imgs, self.B_paths = store_dataset(self.dir_B)

        # self.A_paths = sorted(self.A_paths)
        # self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform = get_transform(opt)

    def __getitem__(self, index):

        idx_A = np.random.randint(0, self.A_size-1)
        idx_B = np.random.randint(0, self.B_size-1)
        #idx_B_OverExp = np.random.randint(0, self.B_size-1)
        
        A_img = self.A_imgs[idx_A]
        B_img = self.B_imgs[idx_B]
        #B_img_overexp = self.B_imgs[idx_B_OverExp]
        A_path = self.A_paths[idx_A]
        B_path = self.B_paths[idx_B]

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        A_gray = torch.unsqueeze(A_img[0], 0)
        input_img = A_img

        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path, 'enhance_level_A2B': torch.rand(A_gray.shape[0], 1, 1),
                'enhance_level_B2A': torch.rand(A_gray.shape[0], 1, 1)}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedDataset'


