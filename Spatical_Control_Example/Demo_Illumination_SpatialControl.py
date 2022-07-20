import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from util.MonoPix_utils import *
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()

config_path = 'configs/MonoPix_LowLight_Default.yaml' # Cyc_CatDog_UnetTanh_BasicD_NoNormG_IPBug_CYC-133

print('loading from config')
f = yaml.safe_load(open(config_path, 'r'))
for kk, vv in f.items():
    setattr(opt, kk, vv)

# resume
setattr(opt, 'resume_ckpt', 1)
setattr(opt, 'which_direction', 'BtoA')
setattr(opt, 'batchSize', 8)
setattr(opt, 'which_epoch', 130)
setattr(opt, 'debug', 1)
setattr(opt, 'gpu_ids', [0])
setattr(opt, 'isTrain', 0)
setattr(opt, 'vis_IN', 0)
setattr(opt, 'pretrained_path', './checkpoints/' + config_path[:-4])  # './ckpt_Ctrl_PreT/Reweight_G' './checkpoints/CEG_LOLS_VGG1_FT_FewShot'
print('--------args----------')
for k in list(sorted(vars(opt).keys())):
    print('%s: %s' % (k, vars(opt)[k]))
print('--------args----------\n')

# seeds
def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
init_seeds(opt.seeds)

# save to the disk
name = 'LL_Multiple'  #  SummWint CatDog  Gender
os.makedirs('./visualize_demo/'+name, exist_ok=True)
model = create_model(opt)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# TEST OK
# LL-test
# user-defined continuous spatial lvl
img_name = './Spatical_Control_Example/low10450.png'
img = io.imread(img_name)
img = img[0:400,100:500,:]
spatial_temp = get_spatial_continuous(img.shape, min_=-3.0, max_=1,rev=True)
spatial_temp = spatial_temp
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=-1.0,max_val=0.5,name='./visualize_demo/LL_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='LL_continuous_h2l')



# multiple enhance levels:
img = io.imread('./Spatical_Control_Example/low10458.png')
for i in range(10):
    spatial_temp = get_spatial_continuous(img.shape, min_=i/10, max_=i/10+0.01,rev=False)
    #save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=-1.0,max_val=0.5,name='LL_spatial',verbose=True,size=(4.5,4))
    inp_ = get_special_input_lvlImgSize(img, spatial_temp)
    model.pred_special_single(*inp_, onevariable=False, name='LL_Multiple/LL_{}'.format(i))



# MASK ON Low-light enhancement task
img = io.imread('./Spatical_Control_Example/low10413.png')
imgmask_name = './Spatical_Control_Example/low10413_mask.png'
imgmask_input = io.imread(imgmask_name).reshape(-1,3)
img_mask_digital = np.zeros(imgmask_input.shape[0])
for idx, pixel in enumerate(imgmask_input):
    if float(pixel[0])>250 and float(pixel[1])>250 and float(pixel[2])>250:  # white
        img_mask_digital[idx]=0.7
    elif float(pixel[0])>250 and float(pixel[1])<50 and float(pixel[2])<50: # red
        img_mask_digital[idx]=-0.5
    else:
        img_mask_digital[idx] = -2
img_mask_digital = img_mask_digital.reshape(img.shape[0], img.shape[1])
#io.imsave('digital_mask_S2W.png', img_mask_digital)
#spatial_temp = get_strip_lvl(img.shape, num_strips=5, min_=0, max_=1)
save_singleChannel_jit(img_mask_digital,min_val=-1,max_val=0.7,name='./visualize_demo/Flower_spatial',verbose=False,size=(4.5,4))
inp_ = get_special_input_lvlImgSize(img, torch.tensor(img_mask_digital, dtype=torch.float32).cuda(0))
model.pred_special_single(*inp_, onevariable=False, name='LOL_amsked')



# TEST OK
# SIDD EVAL SPATIAL
img_name = './Spatical_Control_Example/low10461.png'
img = io.imread(img_name)
# img =img.
spatial_temp = get_strip_lvl(img.shape, num_strips=5, min_=0, max_=1)
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=1,name='./visualize_demo/LL_strips',verbose=True)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='LL_spatial_test')
