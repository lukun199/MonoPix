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


class LowlightDataset(BaseDataset):
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
        # A_path = self.A_paths[index % self.A_size]
        # B_path = self.B_paths[index % self.B_size]
        
        # A_img = Image.open(A_path).convert('RGB')
        # B_img = Image.open(B_path).convert('RGB')
        idx_A = np.random.randint(0, self.A_size-1)
        idx_B = np.random.randint(0, self.B_size-1)
        #idx_B_OverExp = np.random.randint(0, self.B_size-1)
        
        A_img = self.A_imgs[idx_A]
        B_img = self.B_imgs[idx_B]
        #B_img_overexp = self.B_imgs[idx_B_OverExp]
        A_path = self.A_paths[idx_A]
        B_path = self.B_paths[idx_B]
        # A_size = A_img.size
        # B_size = B_img.size
        # A_size = A_size = (A_size[0]//16*16, A_size[1]//16*16)
        # B_size = B_size = (B_size[0]//16*16, B_size[1]//16*16)
        # A_img = A_img.resize(A_size, Image.BICUBIC)
        # B_img = B_img.resize(B_size, Image.BICUBIC)
        # A_gray = A_img.convert('LA')
        # A_gray = 255.0-A_gray

        A_img = self.transform(A_img)
        """ not used in the paper.
        if not self.opt.usePreGen:
            if random.random() < self.opt.dataset_mode_prob:
                if self.opt.target_mode == 'more_over':
                    B_img = transforms.ColorJitter(brightness=(1., 2))(B_img)
                elif self.opt.target_mode == 'more_under':
                    B_img = transforms.ColorJitter(brightness=(0.5, 1))(B_img)
                else:
                    pass
        """
        B_img = self.transform(B_img)
        

        #print('target jitter')  # transform only for 0-255
        #B_img_overexp = transforms.ColorJitter(brightness=(1.2, 3))(B_img_overexp)
        #B_img_overexp = self.transform(B_img_overexp)
        
        
        if self.opt.resize_or_crop == 'no':
            r,g,b = A_img[0]+1, A_img[1]+1, A_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
            input_img = A_img
            # A_gray = (1./A_gray)/255.
        else:
            w = A_img.size(2)
            h = A_img.size(1)
            
            # A_gray = (1./A_gray)/255.
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(2, idx)
                B_img = B_img.index_select(2, idx)
                #B_img_overexp = B_img_overexp.index_select(2, idx)
            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A_img.size(1) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A_img = A_img.index_select(1, idx)
                B_img = B_img.index_select(1, idx)
                #B_img_overexp = B_img_overexp.index_select(1, idx)
            if self.opt.vary == 1 and (not self.opt.no_flip) and random.random() < 0.5:
                times = random.randint(self.opt.low_times,self.opt.high_times)/100.
                input_img = (A_img+1)/2./times
                input_img = input_img*2-1
            else:
                input_img = A_img
            if self.opt.lighten:
                B_img = (B_img + 1)/2.
                B_img = (B_img - torch.min(B_img))/(torch.max(B_img) - torch.min(B_img))
                B_img = B_img*2. -1
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)
        return {'A': A_img, 'B': B_img, 'A_gray': A_gray, 'input_img': input_img,
                'A_paths': A_path, 'B_paths': B_path, 'enhance_level': torch.rand(A_gray.shape[0], 1, 1)}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'LowlightDataset'


