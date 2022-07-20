import torchvision.models as models
import torch, os, argparse, json
import torch.nn as nn
from skimage import io, transform
import numpy as np
import pickle
from scipy import linalg

class InceptionV3(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        return x


# https://github.com/clovaai/stargan-v2/blob/master/metrics/fid.py
def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--gt_path', type=str, default='')
parser.add_argument('--cal_DomainA', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--start_lvl', type=int, default=0)
parser.add_argument('--savefile', type=int, default=1)
args = parser.parse_args()

print('--------args----------')
for k in list(sorted(vars(args).keys())):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

device = torch.device(args.device)
model = InceptionV3()
model.eval()
model.to(device)

WholeTraj = args.start_lvl==0

if WholeTraj:
    if 'S2W' in args.input_path or 'W2S' in args.input_path or 'SW' in args.input_path:
        style = 'Yosemite'
    else:
        style = 'AFHQ'
else:
    if 'S2W' in args.input_path or 'W2S' in args.input_path or 'SW' in args.input_path:
        style = 'Winter' if args.cal_DomainA > 0.5 else 'Summer'  # if cal_Domain A, the target feats are from domain A
    else:
        style = 'Cat' if args.cal_DomainA > 0.5 else 'Dog'


def norm_meanstd(vector, mean, std):
    vector/=255.
    mean.reshape(1,1,-1)
    std.reshape(1, 1, -1)
    return (vector-mean)/std

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

pkl_file_path = os.path.join(os.path.dirname(__file__), 'pkl_feats')
os.makedirs(pkl_file_path, exist_ok=True)

A_DOMAIN = 'A_SW' if 'S2W' in args.input_path or 'W2S' in args.input_path or 'SW' in args.input_path else 'ACat'
B_DOMAIN = 'B_SW' if 'S2W' in args.input_path or 'W2S' in args.input_path or 'SW' in args.input_path else 'BDog'

pkl_file_name = os.path.join(pkl_file_path, 'feats_{}_startlvl_{}_{}.pkl'.format(style, args.start_lvl,
                                                                         'wholetraj' if WholeTraj else ''))

if not os.path.exists(pkl_file_name):
    # save gt feats from dataset, as they are shared by all methods.
    with torch.no_grad():
        actvs = []
        A_PATH = os.path.join(args.gt_path, 'train' + A_DOMAIN)
        B_PATH = os.path.join(args.gt_path, 'train' + B_DOMAIN)
        if WholeTraj:
            all_img_names = [os.path.join(A_PATH, x) for x in os.listdir(A_PATH)] + \
                            [os.path.join(B_PATH, x) for x in os.listdir(B_PATH)]
        else:
            all_img_names = [os.path.join(A_PATH, x) for x in os.listdir(A_PATH)] if args.cal_DomainA>0.5 \
                            else [os.path.join(B_PATH, x) for x in os.listdir(B_PATH)]

        print('-----[*]---len of imgs: ', len(all_img_names))
        for img_name in all_img_names:
            img = io.imread(img_name)
            # ImageNet norm
            img = transform.resize(img, (299, 299), anti_aliasing=True, preserve_range=True)
            img = norm_meanstd(img, mean, std)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            actv = model.forward(img)
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()  # [N, 2048]
        gt_mu = np.mean(actvs, axis=0)  # [1, 2048]
        gt_cov = np.cov(actvs, rowvar=False)
        with open(pkl_file_name, 'wb') as f:
            pickle.dump({'mu': gt_mu, 'cov': gt_cov}, f)
else:
    # load corresponding gt_feats
    with open(pkl_file_name, 'rb') as f:
         gt_feats = pickle.load(f)
         gt_mu, gt_cov = gt_feats['mu'], gt_feats['cov']
    print('-------------load {}'.format(pkl_file_name))

with torch.no_grad():
    feas_all = []
    for this_folder in sorted(os.listdir(args.input_path)):
        if '.' in this_folder:
            continue
        this_folder_path = os.path.join(args.input_path, this_folder)
        imgs = []
        for idx in range(args.start_lvl, 11):  # lvl: [00, 01, 02, ... 10]
            img = io.imread(this_folder_path + '/{}_enhlvl_{:02d}.png'.format(this_folder, idx))
            # ImageNet norm
            img = transform.resize(img, (299, 299), anti_aliasing=True, preserve_range=True)
            img = norm_meanstd(img, mean, std)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            imgs.append(img)
        imgs = torch.stack(imgs)
        imgs = imgs.to(device)

        res_feats = model.forward(imgs)
        res_feats = res_feats.cpu().numpy()
        feas_all.append(res_feats)  # [N, 11, ...]

    feas_all = np.vstack(feas_all)
    feas_mu = np.array(feas_all).mean(0)
    res_this = frechet_distance(gt_mu, gt_cov, feas_mu, np.cov(feas_all, rowvar=False))

if args.savefile>0:
    with open(os.path.join(args.input_path, 'FID_{}_startlvl_{}_{}.json'.format(style, args.start_lvl,
                                                     'wholetraj' if WholeTraj else '')), 'w') as f:
        json.dump({'FID_all':res_this}, f)
