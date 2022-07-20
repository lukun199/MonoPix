import os
import matplotlib.pyplot as plt
import argparse
from skimage import io
import numpy as np
import json
from scipy.spatial.distance import jensenshannon as JS_Div


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='')
args = parser.parse_args()

Exp_path = args.path

target_lpips_file = os.path.join(Exp_path, 'lpips_all_.json')

with open(target_lpips_file, 'r') as f:
    data = json.load(f)

adj_all = []
against1_all = []

for key_this in sorted(data.keys()):  # key is the img index
    dat_this = data[key_this]
    data_this_adj, data_this_against1 = dat_this['adj'], dat_this['against1']
    adj_all.append(data_this_adj)
    against1_all.append(data_this_against1)

# [num_of_imgs, ~10]
against1_all = np.array(against1_all)
adj_all = np.array(adj_all)

AbsoluteLinearity = np.mean([np.corrcoef(np.arange(0, 11), x)[0, 1] for x in against1_all])
Smooth = np.mean(np.max(adj_all, -1))
Range = [max(x) - min(x) for x in against1_all]
RelativeLinearity = np.mean([np.corrcoef(np.arange(0, 10), x.cumsum())[0, 1] for x in adj_all])

dict_continuity = {
    'Range': np.mean(Range).item(),
    'Smoothness': Smooth.item(),
    'AbsoluteLinearity': AbsoluteLinearity,
    'RelativeLinearity': RelativeLinearity
}

with open(os.path.join(Exp_path, 'Continuity_Metric.json'), 'w') as f:
    json.dump(dict_continuity, f)