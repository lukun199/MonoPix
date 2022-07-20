import torchvision.models as models
import torch, os, argparse, json
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms as T
from PIL import Image
import random
from skimage import io
import numpy as np

# cal the overall quality (maximum).
class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.resnet50 = models.resnet50(num_classes=64)
        self.fcs = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(),
                                 nn.Linear(64, 32), nn.LeakyReLU(),
                                 nn.Linear(32, 1))

    def forward(self, x):
        x = self.resnet50(x)
        x = self.fcs(x)
        return x

def cal_domainA_score(model_, x):
    return torch.nn.functional.sigmoid(model_(x))


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='')
parser.add_argument('--cal_DomainA', type=float, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seeds', type=int, default=0)
parser.add_argument('--ckpt_loadpath', type=str, default=os.path.join(os.path.dirname(__file__),'checkpoints','ACC','SW','epoch50.pth'),
                    help='SW-ckpt50; or CD-ckpt5')
args = parser.parse_args()


model = Mymodel()
model.eval()
device = torch.device(args.device)
model.load_state_dict(torch.load(args.ckpt_loadpath, map_location=args.device if 'cuda' in args.device else 'cpu'))

model.to(device)


with torch.no_grad():
    res_all = []
    for this_folder in sorted(os.listdir(args.input_path)):
        if '.' in this_folder:
            continue
        this_folder_path = os.path.join(args.input_path, this_folder)
        imgs = []
        for idx in range(11):
            img = io.imread(this_folder_path + '/{}_enhlvl_{:02d}.png'.format(this_folder, idx))
            img = (img / 255.0 - 0.5) * 2
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
            imgs.append(img)
        imgs = torch.stack(imgs)
        imgs = imgs.to(device)

        # during training, we label Winter/Cat (domain A) as 1; Summer/Dog (domain B) is 0
        res_Conf = cal_domainA_score(model, imgs)
        if args.cal_DomainA < 0.5:  # we only have two classes
            res_Conf = 1 - res_Conf
        res_Conf = res_Conf.cpu().numpy()

        with open(this_folder_path + '/score_this.json', 'w') as f:
            json.dump({'score_this':res_Conf.tolist()}, f)

        res_all.append(np.max(res_Conf).item())

with open(args.input_path + '/Score_ACC{}.json'.format(os.path.basename(args.ckpt_loadpath).split('_')[0][5:]), 'w') as f:
    json.dump({'score_all':res_all, 'score_mean':np.mean(res_all).item()}, f)


""" This is how we trained the model.
import torchvision.models as models
import torch, os, argparse
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms as T
from PIL import Image
import random

parser = argparse.ArgumentParser()
parser.add_argument('--A_path', type=str, default='trainA_SW')  # Winter/Cat
parser.add_argument('--B_path', type=str, default='trainB_SW')  # Summer/Dog
parser.add_argument('--A_path_test', type=str, default='testA_SW')
parser.add_argument('--B_path_test', type=str, default='testB_SW')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--batchSize', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--total_epoches', type=int, default=101)
parser.add_argument('--save_per_epoch', type=int, default=10)
parser.add_argument('--name', type=str, default='SW')
parser.add_argument('--seeds', type=int, default=0)

args = parser.parse_args()
os.makedirs('./checkpoints/{}'.format(args.name), exist_ok=True)

print('--------args----------')
for k in list(sorted(vars(args).keys())):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
init_seeds(args.seeds)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def store_dataset(dir):
    images = []
    all_path = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img = Image.open(path).convert('RGB')
                img = img.resize((256, 256))
                images.append(img)
                all_path.append(path)
    return images, all_path


class dataset_single(data.Dataset):
  def __init__(self, path, args):
    self.dir_A = os.path.join(path, args.A_path)
    self.dir_B = os.path.join(path, args.B_path)
    self.A_imgs, self.A_paths = store_dataset(self.dir_A)
    self.B_imgs, self.B_paths = store_dataset(self.dir_B)
    self.A_size = len(self.A_paths)
    self.B_size = len(self.B_paths)
    print('A_size: {},  B_size{}'.format(self.A_size, self.B_size))
    # setup image transformation
    transforms = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]))
    self.transforms = T.Compose(transforms)

  def __getitem__(self, index):
    if index>self.A_size-1:
        img = self.B_imgs[index-self.A_size]  # B
        label = torch.tensor(0)
    else:
        img = self.A_imgs[index]  # A
        label = torch.tensor(1)
    img = self.transforms(img)

    return img, label

  def __len__(self):
    return self.A_size + self.B_size



class dataset_test(data.Dataset):
  def __init__(self, path, args):
    self.dir_A = os.path.join(path, args.A_path_test)
    self.dir_B = os.path.join(path, args.B_path_test)
    self.A_imgs, self.A_paths = store_dataset(self.dir_A)
    self.B_imgs, self.B_paths = store_dataset(self.dir_B)
    self.A_size = len(self.A_paths)
    self.B_size = len(self.B_paths)
    print('A_size: {},  B_size{}'.format(self.A_size, self.B_size))
    # setup image transformation
    transforms = []
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]))
    self.transforms = T.Compose(transforms)

  def __getitem__(self, index):
    if index>self.A_size-1:  # B is labeled as 0
        img = self.B_imgs[index-self.A_size]
        label = torch.tensor(0)
    else:
        img = self.A_imgs[index]
        label = torch.tensor(1)
    img = self.transforms(img)

    return img, label

  def __len__(self):
    return self.A_size + self.B_size

class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.resnet50 = models.resnet50(num_classes=64)
        self.fcs = nn.Sequential(nn.Linear(64, 64), nn.LeakyReLU(),
                                  nn.Linear(64, 32), nn.LeakyReLU(),
                                  nn.Linear(32, 1))
    def forward(self, x):
        x = self.resnet50(x)
        x = self.fcs(x)
        return x

def run_test(A_size):
    cnt_A = cnt_B = 0
    total = 0
    for idx, data in enumerate(data_loader_test):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.float().to(device).view(-1,1)
        res = model(imgs)
        if labels[0] > 0.5 and res.item()>0:  # belong to A
            if idx<A_size:
                cnt_A+=1
            else:
                cnt_B+=1
        elif labels[0] < 0.5 and res.item()<0:  # belong to B
            if idx<A_size:
                cnt_A += 1
            else:
                cnt_B+=1
        total += 1
    print('right num A: {}/{}'.format(cnt_A, A_size))
    print('right num B: {}/{}'.format(cnt_B, total-A_size))
    print('all_right: ', cnt_A+cnt_B)


device = torch.device(args.device)
dataset = dataset_single(args.data_path, args)
data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batchSize,
                shuffle=True,
                num_workers=4,
                )
dataset_test = dataset_test(args.data_path, args)
data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                )

crit = nn.BCEWithLogitsLoss()

model = Mymodel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
model.train()


for epoch in range(1, args.total_epoches):
    for idx, data in enumerate(data_loader):
        imgs, labels = data
        imgs = imgs.to(device)
        labels = labels.float().to(device).view(-1,1)
        res = model(imgs)
        loss = crit(res, labels)
        print('[{}/{}] loss = {}'.format(idx, epoch, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        model.eval()
        run_test(dataset_test.A_size)
        model.train()
        
    if epoch%args.save_per_epoch==0:
        torch.save(model.state_dict(), './checkpoints/{}/epoch{}_resnet50.pth'.format(args.name, epoch))

"""