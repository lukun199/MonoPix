import os
import re
from collections import OrderedDict
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        saved_ckpts = torch.load(save_path)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            print('the ckpts are not consistent. may be the framework is changed.')
            print('trying to rename all norm layers')
            _dict = OrderedDict()
            bn_pattern = re.compile('bn\d_\d')
            for kk, vv in saved_ckpts.items():
                prefix = bn_pattern.findall(kk)
                if prefix:
                    _dict['norm'+kk[2:]]=vv  # change 'bn' to 'norm'
                else:
                    _dict[kk]=vv
            network.load_state_dict(_dict)
        print('[*]-------------successfully loaded ckpts.')

    def update_learning_rate():
        pass
