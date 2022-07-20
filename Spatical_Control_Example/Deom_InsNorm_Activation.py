import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io, transform
from util.MonoPix_utils import *
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()

config_path = 'configs/MonoPix_SummerWinter_InsNorm.yaml'

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
setattr(opt, 'vis_IN', 1)  # when show vis.
setattr(opt, 'pretrained_path', './checkpoints/' + config_path[:-4])  # './ckpt_Ctrl_PreT/Reweight_G' './checkpoints/CEG_LOLS_VGG1_FT_FewShot'
print('--------args----------')
for k in list(sorted(vars(opt).keys())):
    print('%s: %s' % (k, vars(opt)[k]))
print('--------args----------\n')

def relu(arr):
    return np.maximum(0.2 * arr, arr)

def norm(arr):
    return (arr-np.mean(arr)) / np.std(arr)

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

os.makedirs('./visualize_demo/kde_example',exist_ok=True)
os.makedirs('./visualize_demo/vis_IN_ACT',exist_ok=True)
model = create_model(opt)


import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def vis_IN_Act_float(arr, name):
    proto = np.zeros_like(arr[0][1])
    arr = np.maximum(0.2 * arr, arr)
    # arr[0] = np.where(arr[0] < 0, arr[0], arr[0] * 0.2)
    #sc = plt.imshow(arr[1].mean(0), cmap=plt.cm.jet)# interpolation="nearest",
    sc = plt.imshow(np.abs(arr[1].mean(0)), cmap=plt.cm.jet)#  vmin=0.06, vmax=0.13) # [0, 0.1] for WS. [0.16, .0.2] for SW
    #plt.colorbar()
    plt.axis('off')
    sc.figure.savefig('./visualize_demo/vis_IN_ACT/' + name.split('.')[0] + '_vis_floatACT.png', bbox_inches='tight',pad_inches = 0, dpi=60)
    plt.clf()


def vis_IN_Act_count(arr, name):
    proto = np.zeros_like(arr[0][1])
    for i, x in enumerate(arr[1] - arr[0]):
        if x.mean() > 0:
            proto[np.where(arr[0][i] < 0)] += 1
        else:
            proto[np.where(arr[0][i] > 0)] += 1
    sc = plt.imshow(proto, cmap=plt.cm.jet)# interpolation="nearest",
    plt.axis('off')
    #sc = plt.pcolor(arr[1].mean(0), cmap=plt.cm.jet, vmin=0, vmax=0.1)
    #plt.colorbar()
    sc.figure.savefig('./visualize_demo/vis_IN_ACT/' + name.split('.')[0] + '_vis_CountACT.png', bbox_inches='tight',pad_inches = 0, dpi=60)
    plt.clf()


def vis_IN_Act_count_crossZero(arr, name):
    proto = np.zeros_like(arr[0][1])
    for i, x in enumerate(arr[1] - arr[0]):
        #if x.mean() > 0:
        set0 = set(zip(*np.where(arr[0][i] < 0)))
        set1 = set(zip(*np.where(arr[1][i] > 0)))
        pos = list(zip(*(set0&set1)))
        proto[pos] += 1
        #else:
        set0 = set(zip(*np.where(arr[0][i] > 0)))
        set1 = set(zip(*np.where(arr[1][i] < 0)))
        pos = list(zip(*(set0&set1)))
        proto[pos] += 1
    sc = plt.imshow(proto, cmap=plt.cm.jet, vmax=3)# interpolation="nearest",
    plt.axis('off')
    #sc = plt.pcolor(arr[1].mean(0), cmap=plt.cm.jet, vmin=0, vmax=0.1)
    #plt.colorbar()
    sc.figure.savefig('./visualize_demo/vis_IN_ACT/' + name.split('.')[0] + '_vis_CountACT_CrossZero.png', bbox_inches='tight', pad_inches = 0, dpi=60)
    plt.clf()



# Used to visualize the IN attention

inner_name = './Spatical_Control_Example/54_enhlvl_99.png'
inp_ = get_special_input_Single(inner_name, device=torch.device("cuda:{}".format(opt.gpu_ids[0])),start=0, end=1.0)
res, vis = model.pred_special_test(*inp_)
vis_IN_Act_float(vis[[0,-1]].cpu().numpy(), inner_name.split('\\')[-1])
vis_IN_Act_count(vis[[0,-1]].cpu().numpy(), inner_name.split('\\')[-1])
vis_IN_Act_count_crossZero(vis[[0,-1]].cpu().numpy(), inner_name.split('\\')[-1])


# plot the distribution figure.
import seaborn as sns

c1, c2 = sns.color_palette("husl", 3)[:2]

dis_data0 = []
dis_data1 = []
for x in vis[0].cpu().numpy():
   dis_data0.append(x[10:-10, 10:-10].reshape(-1))
for x in vis[-1].cpu().numpy():
    dis_data1.append(x[10:-10, 10:-10].reshape(-1))


fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# if dis_data0[i].max() * dis_data0[i].min() <0:
filter_idx = 25
sns.kdeplot(dis_data0[filter_idx], shade=True, color=c1, ax=ax, bw_adjust=.2)
sns.kdeplot(dis_data1[filter_idx], shade=True, color=c2, ax=ax, bw_adjust=.2)
plt.yticks([])
fig.savefig('./visualize_demo/kde_example/kde_plot_input_{}.png'.format(filter_idx), bbox_inches='tight',pad_inches=0)
fig.clf()


fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# if dis_data0[i].max() * dis_data0[i].min() <0:
filter_idx = 25
sns.kdeplot(relu(dis_data0[filter_idx]), shade=True, color=c1, ax=ax, bw_adjust=.2)
sns.kdeplot(relu(dis_data1[filter_idx]), shade=True, color=c2, ax=ax, bw_adjust=.2)
plt.yticks([])
fig.savefig('./visualize_demo/kde_example/kde_plot_AfterReLu_{}.png'.format(filter_idx), bbox_inches='tight',pad_inches=0)
fig.clf()


fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# if dis_data0[i].max() * dis_data0[i].min() <0:
input0 = norm(relu(dis_data0[filter_idx]))
input1 = norm(relu(dis_data1[filter_idx]))
sns.kdeplot(input0, shade=True, color=c1, ax=ax, bw_adjust=.2)
sns.kdeplot(input1, shade=True, color=c2, ax=ax, bw_adjust=.2)
plt.yticks([])
fig.savefig('./visualize_demo/kde_example/kde_plot_test_IN-{}.png'.format(filter_idx), bbox_inches='tight',pad_inches=0)
fig.clf()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# if dis_data0[i].max() * dis_data0[i].min() <0:
input0 = norm(dis_data0[filter_idx])
input1 = norm(dis_data1[filter_idx])
sns.kdeplot(input0, shade=True, color=c1, ax=ax, bw_adjust=.2)
sns.kdeplot(input1, shade=True, color=c2, ax=ax, bw_adjust=.2)
plt.yticks([])
fig.savefig('./visualize_demo/kde_example/kde_plot_test_DirectIN-{}.png'.format(filter_idx), bbox_inches='tight',pad_inches=0)
fig.clf()
