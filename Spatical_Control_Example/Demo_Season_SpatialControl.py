import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io, transform
from util.MonoPix_utils import *
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()

config_path = 'configs/MonoPix_SummerWinter_Default.yaml'

print('loading from config')
f = yaml.safe_load(open(config_path, 'r'))
for kk, vv in f.items():
    setattr(opt, kk, vv)

# resume
setattr(opt, 'resume_ckpt', 1)
setattr(opt, 'which_direction', 'BtoA')
setattr(opt, 'batchSize', 8)
setattr(opt, 'which_epoch', 300)
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


os.makedirs('./visualize_demo', exist_ok=True)
model = create_model(opt)


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


img_name = './Spatical_Control_Example/2013-07-11 11_43_11.jpg'
img = io.imread(img_name)#[50:250, 50:250,:]
spatial_temp = get_spatial_continuous(img.shape, min_=-0.5, max_=1.5,rev=False)
spatial_temp = spatial_temp.T
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0,max_val=1,name='./visualize_demo/S2W_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='S2W_continuous_l2hT', dir='BtoA')


# TEST OK
# W2S_MASKED
# user-defined strip lvl
img_name = './Spatical_Control_Example/32_enhlvl_99.png'
img = io.imread(img_name)
imgmask_name = './Spatical_Control_Example/32_enhlvl_99_maskhalf.png'
imgmask_input = io.imread(imgmask_name).reshape(-1,3)
img_mask_digital = np.zeros(imgmask_input.shape[0])
for idx, pixel in enumerate(imgmask_input):
    if float(pixel[0])>250 and float(pixel[1])>250 and float(pixel[2])>250:
        img_mask_digital[idx]=1
img_mask_digital = img_mask_digital.reshape(img.shape[0], img.shape[1])
#io.imsave('digital_mask_S2W.png', img_mask_digital)
#spatial_temp = get_strip_lvl(img.shape, num_strips=5, min_=0, max_=1)
save_singleChannel_jit(img_mask_digital,min_val=0,max_val=1,name='./visualize_demo/S2W_spatial2',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, torch.tensor(img_mask_digital, dtype=torch.float32).cuda(0))
model.pred_special_single(*inp_, onevariable=False, name='S2W_masked2')
