import lpips
import os
import torch
import argparse
from skimage import io
import numpy as np
import json

def topn1(img):
    # lpips accepts [-1, +1] as the input
    # https://github.com/richzhang/PerceptualSimilarity
    # input: [0, 255]; output:[-1, +1]
    return ((img/255.)-0.5)*2

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

device = torch.device(args.device)

lpips_calfun = lpips.LPIPS(net='alex').to(device)

# we assume each image has a folder. the images are named 'xx_00.png'
all_imgs_folder_path = args.path
lpips_all = {}

for this_image_folder in os.listdir(all_imgs_folder_path):
    if '.' in this_image_folder: continue
    this_path = os.path.join(all_imgs_folder_path, this_image_folder)
    lpips_adj = []
    lpips_against1 = []
    lpips_this = {}
    img_0 = io.imread(os.path.join(this_path, this_image_folder+'_enhlvl_99.png'))  # this is the input image
    for i in range(10): # 0-9
        img_1 = io.imread(os.path.join(this_path, this_image_folder+'_enhlvl_{:02d}.png'.format(i)))
        img_2 = io.imread(os.path.join(this_path, this_image_folder + '_enhlvl_{:02d}.png'.format(i+1)))
        lpips_adj_score = lpips_calfun(torch.FloatTensor(topn1(img_2).transpose(2,0,1)).unsqueeze(0).to(device),
                                 torch.FloatTensor(topn1(img_1).transpose(2,0,1)).unsqueeze(0).to(device))
        lpips_adj.append(lpips_adj_score.item())
    for i in range(11):  # 0-10
        img_3 = io.imread(os.path.join(this_path, this_image_folder + '_enhlvl_{:02d}.png'.format(i)))
        lpips_aginst1_score = lpips_calfun(
            torch.FloatTensor(topn1(img_0).transpose(2, 0, 1)).unsqueeze(0).to(device),
            torch.FloatTensor(topn1(img_3).transpose(2, 0, 1)).unsqueeze(0).to(device))
        lpips_against1.append(lpips_aginst1_score.item())

    lpips_this['adj'] = lpips_adj
    lpips_this['against1'] = lpips_against1

    with open(os.path.join(this_path, 'lpips_this.json'), 'w') as f:
        json.dump({'this_lpips': lpips_this}, f)
    lpips_all[this_image_folder] = lpips_this

with open(os.path.join(all_imgs_folder_path, 'lpips_all_.json'), 'w') as f:
    json.dump(lpips_all, f)
