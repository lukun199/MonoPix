import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io, transform
from util.MonoPix_utils import *
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()

config_path = 'configs/MonoPix_Noise_Default.yaml'

print('loading from config')
f = yaml.safe_load(open(config_path, 'r'))
for kk, vv in f.items():
    setattr(opt, kk, vv)

# resume
setattr(opt, 'resume_ckpt', 1)
setattr(opt, 'which_direction', 'BtoA')
setattr(opt, 'batchSize', 8)
setattr(opt, 'which_epoch', 60)
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
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
init_seeds(opt.seeds)

model = create_model(opt)


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


# TEST OK
# SIDD EVAL SPATIAL



img_name = './Spatical_Control_Example/12.png'
img = io.imread(img_name)
img = img[:255, :255,:]
spatial_temp = get_strip_lvl(img.shape, num_strips=3, min_=0.5, max_=1.0)
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0.5,max_val=1,name='./visualize_demo/SIDD_spatial',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='SIDD_spatial_l2h')


# figures in the supp.
img_name = './Spatical_Control_Example/904.png'
img = io.imread(img_name)
spatial_temp = get_spatial_continuous(img.shape, min_=0.3, max_=1,rev=True)
#spatial_temp = spatial_temp.T
save_singleChannel_jit(spatial_temp.cpu().numpy(),min_val=0.3,max_val=1,name='./visualize_demo/SIDD_spatial_supp',verbose=True,size=(4.5,4), dpi=60)
inp_ = get_special_input_lvlImgSize(img, spatial_temp)
model.pred_special_single(*inp_, onevariable=False, name='SIDD_spatial_h2l')
