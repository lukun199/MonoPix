import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io, transform
from util.MonoPix_utils import *
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()

config_path = 'configs/MonoPix_CatDog_Default.yaml'

print('loading from config')
f = yaml.safe_load(open(config_path, 'r'))
for kk, vv in f.items():
    setattr(opt, kk, vv)

# resume
setattr(opt, 'resume_ckpt', 1)
setattr(opt, 'which_direction', 'BtoA')
setattr(opt, 'batchSize', 8)
setattr(opt, 'which_epoch', 200)
setattr(opt, 'debug', 1)
setattr(opt, 'gpu_ids', [0])
setattr(opt, 'isTrain', 0)
setattr(opt, 'vis_IN', 0)
setattr(opt, 'pretrained_path', './checkpoints/' + config_path[:-4])
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
name = 'CatDog'  #  SummWint CatDog  Gender
os.makedirs('./visualize_demo/'+name, exist_ok=True)
model = create_model(opt)


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# the is following used for fine-grained GLOBAL lvl control.
# Though not linear enough, it is monotonic and continuous
inner_name = r'./Spatical_Control_Example/105_enhlvl_99.png'
inp_ = get_special_input_Single(inner_name, start=0, end=1, device=torch.device("cuda:{}".format(opt.gpu_ids[0])))
res, vis =  model.pred_special_test(*inp_)
for idx, res_ in enumerate(res):
    res_ = np.transpose(res_, (1, 2, 0))
    res_ = np.minimum(res_, 1)
    res_ = np.maximum(res_, -1)
    io.imsave('./visualize_demo/' + '/CatDog/{}_{:02d}.png'.format('IOI_105_enhlvl', idx),
              ((res_ + 1) / 2 * 255.).astype(np.uint8))


img_name = './Spatical_Control_Example/56_enhlvl_99.png'
img = io.imread(img_name)
spatial_temp = get_spatial_continuous(img.shape, min_=0, max_=0.8,rev=True)
spatial_temp = spatial_temp
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=0.8,name='./visualize_demo/D2C_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='D2C_continuous_h2l', dir='BtoA')

img_name = './Spatical_Control_Example/56_enhlvl_99.png'
img = io.imread(img_name)
spatial_temp = get_spatial_continuous(img.shape, min_=0, max_=0.8,rev=True)
spatial_temp = spatial_temp.T
#save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=0.65,name='./visualize_demo/D2C_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='D2C_continuous_h2lT', dir='BtoA')

img_name = './Spatical_Control_Example/56_enhlvl_99.png'
img = io.imread(img_name)
spatial_temp = get_spatial_continuous(img.shape, min_=0, max_=0.8,rev=False)
spatial_temp = spatial_temp
#save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=0.8,name='./visualize_demo/D2C_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='D2C_continuous_l2h', dir='BtoA')

img_name = './Spatical_Control_Example/56_enhlvl_99.png'
img = io.imread(img_name)
spatial_temp = get_spatial_continuous(img.shape, min_=0, max_=0.8,rev=False)
spatial_temp = spatial_temp.T
#save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=0.7,name='./visualize_demo/D2C_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='D2C_continuous_l2hT', dir='BtoA')


# 0.7 0 0.3
img_name = './Spatical_Control_Example/16_enhlvl_99.png'
img = io.imread(img_name)
imgmask_name = './Spatical_Control_Example/16_mask.png'
imgmask_input = io.imread(imgmask_name).reshape(-1,3)
img_mask_digital = np.zeros(imgmask_input.shape[0])
for idx, pixel in enumerate(imgmask_input):
    if float(pixel[0])>250 and float(pixel[1])>250 and float(pixel[2])>250:  # white
        img_mask_digital[idx]=0.9
    elif float(pixel[0])<10 and float(pixel[1])<10 and float(pixel[2])<10: # black
        img_mask_digital[idx]=0.15
    else:
        img_mask_digital[idx] = 0.3
img_mask_digital = img_mask_digital.reshape(img.shape[0], img.shape[1])
save_singleChannel_jit(img_mask_digital,min_val=0,max_val=0.9,name='./visualize_demo/D2C_spatial_mask',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, torch.tensor(img_mask_digital, dtype=torch.float32).cuda(0))
model.pred_special_single(*inp_, onevariable=False, name='D2C_masked_face', dir='BtoA')



