from skimage import io
import numpy as np
import torch
import matplotlib.pyplot as plt

# functions in this file are called at the root directory

def get_special_input(name='LL', device=torch.device("cuda:0")):
    if 'SIDD' in name or 'Noise' in name:
        test_img = (io.imread('./visualize/0198_GT_SRGB_010_1.png')/255.-0.5)*2
    elif 'Summ' in name:
        test_img = (io.imread('./visualize/test_Winter.png') / 255. - 0.5) * 2  # test_Winter
    elif 'Cat' in name:
        test_img = (io.imread('./visualize/CD_1.png') / 255. - 0.5) * 2
    else:
        test_img = (io.imread('./visualize/realA_10.png') / 255. - 0.5) * 2


    test_img = np.transpose(test_img, (2,0,1))
    test_bath = np.tile(test_img[np.newaxis], (10,1,1,1))
    r, g, b = test_img[0] + 1, test_img[1] + 1, test_img[2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
    tets_gray = np.tile(A_gray[np.newaxis], (10, 1, 1, 1))
    enhance_level = torch.arange(0, 1, 0.1).view(10, 1, 1, 1).to(device)

    return torch.tensor(test_bath, dtype=torch.float32, device=device), \
       torch.tensor(tets_gray, dtype=torch.float32, device=device), \
        enhance_level


def get_special_input_lvlImgSize(img, enhlvl_spatial, flip=False):
    img_batch = []
    gray_batch = []
    lvl_batch = []

    test_img = (img / 255. - 0.5) * 2
    test_img = np.transpose(test_img, (2, 0, 1))
    r, g, b = test_img[0] + 1, test_img[1] + 1, test_img[2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
    img_batch.append(torch.tensor(test_img, dtype=torch.float32, device=torch.device("cuda:0")))
    gray_batch.append(torch.tensor(A_gray, dtype=torch.float32, device=torch.device("cuda:0")))

    lvl_batch.append(enhlvl_spatial)
    img_batch = torch.stack(img_batch)
    gray_batch = torch.stack(gray_batch)
    lvl_batch = torch.stack(lvl_batch)

    return img_batch, gray_batch.unsqueeze(1), lvl_batch.unsqueeze(1)


def get_special_input_Single(name, start=0, end=1, device=torch.device("cuda:0")):
    test_img = (io.imread(name) / 255. - 0.5) * 2
    test_img = np.transpose(test_img, (2, 0, 1))
    test_bath = np.tile(test_img[np.newaxis], (11, 1, 1, 1))
    r, g, b = test_img[0] + 1, test_img[1] + 1, test_img[2] + 1
    A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
    tets_gray = np.tile(A_gray[np.newaxis], (11, 1, 1, 1))
    enhance_level = torch.linspace(start, end, 11).view(11, 1, 1, 1).to(device)

    return torch.tensor(test_bath, dtype=torch.float32, device=device), \
           torch.tensor(tets_gray, dtype=torch.float32, device=device), \
           enhance_level


def save_singleChannel_jit(numpy_array, min_val=0, max_val=1, name='test_spatial', verbose=False, size=(4, 4), dpi=200):
    # flip the mask to fix the display issue.. though not elegant
    sc = plt.pcolor(np.flipud(numpy_array), cmap=plt.cm.winter, vmin=min_val, vmax=max_val)  # summer
    if verbose:
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
    sc.axes.set_axis_off()
    sc.figure.tight_layout(pad=0)
    sc.figure.set_size_inches(size)
    sc.figure.savefig(name + '.png', dpi=dpi, bbox_inches='tight')
    plt.clf()


def get_spatial_continuous(shape, min_=0, max_=1, rev=False):
    temp = torch.zeros(shape[:2]).to("cuda:0")
    for i in range(shape[1]):
        temp[:, i] = i / (shape[1] / (max_ - min_)) + min_  # 0-1
    if rev:
        temp = temp.fliplr()
    return temp


def get_strip_lvl(shape, num_strips, min_=0, max_=1):
    temp = torch.zeros(shape[:2]).to("cuda:0")
    interval = shape[1] // num_strips
    lvl_interval = np.linspace(min_, max_, num_strips)
    for idx, block_num in enumerate(range(num_strips)):
        temp[:, interval * block_num:interval * block_num + interval] = lvl_interval[idx]
    return temp
