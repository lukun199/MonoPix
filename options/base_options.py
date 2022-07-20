import argparse
import os, platform
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        self.sys_win = 'Windows' in platform.platform()
        
    def initialize(self):
        #self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')  # no use
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')  # no use
        self.parser.add_argument('--patchSize', type=int, default=64, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='sid_unet_resize', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_patchD', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--skip', type=float, default=1.0, help='B = net.forward(A) + skip*A')

        # part of the parameters are introduced by EnlightenGAN, I will format these settings in the future
        self.parser.add_argument('--use_norm', type=float, default=1, help='L1 loss weight is 10.0')  # MonoPix does not use it
        self.parser.add_argument('--use_wgan', type=int, default=0, help='use wgan-gp')  # MonoPix does not use it
        self.parser.add_argument('--vgg', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
        self.parser.add_argument('--vgg_choose', type=str, default='relu5_3', help='choose layer for vgg')
        self.parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
        self.parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
        self.parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
        self.parser.add_argument('--fcn', type=float, default=0, help='use semantic loss')
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--tanh', action='store_true', help='tanh')
        self.parser.add_argument('--out_tanh', type=int, default=1, help='out_tanh in Generator')
        self.parser.add_argument('--linear', action='store_true', help='tanh')  # many parameters are not used in MonoPix
        self.parser.add_argument('--new_lr', action='store_true', help='tanh')
        self.parser.add_argument('--multiply', action='store_true', help='tanh')
        self.parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        self.parser.add_argument('--patchD_3', type=int, default=0, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
        self.parser.add_argument('--times_residual', action='store_true', help='output = input + residual*attention')
        self.parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--high_times', type=int, default=400, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--vary', type=int, default=0, help='use light data augmentation')
        self.parser.add_argument('--lighten', action='store_true', help='normalize attention map')

        # added by MonoPix
        self.parser.add_argument('--grad_clip', type=float, default=0.0)  # optional
        self.parser.add_argument('--Gnorm', type=str, default='Identity')
        self.parser.add_argument('--Dnorm', type=str, default='Identity')
        self.parser.add_argument('--vgg_path', type=str, default='')
        self.parser.add_argument('--dataroot', type=str, default='')
        self.parser.add_argument('--cal_D_Df', type=int, default=0)
        self.parser.add_argument('--cal_G_Df', type=int, default=0)
        self.parser.add_argument('--lambda_mono', type=float, default=1.0)
        self.parser.add_argument('--lambda_df', type=float, default=0.25)
        self.parser.add_argument('--debug', type=int, default=0)
        self.parser.add_argument('--pseudo_best_enlvl', type=float, default=1.0)
        self.parser.add_argument('--custom_lr', action='store_true')
        self.parser.add_argument('--source_dataset', type=str, default='A')
        self.parser.add_argument('--target_dataset', type=str, default='B')
        self.parser.add_argument('--target_mode', type=str, default='normal')
        self.parser.add_argument('--saveD', type=int, default=0)
        self.parser.add_argument('--use_samelvl', type=int, default=0)
        self.parser.add_argument('--idt_anchor', type=float, default=0.0)
        self.parser.add_argument('--monoidt', type=int, default=0)
        self.parser.add_argument('--transcode', type=int, default=0, help='if use more layers before Ins Norm')
        self.parser.add_argument('--vis_IN', type=int, default=0)  # for visualization only
        self.parser.add_argument('--IP_MONO', type=int, default=0)  # set to 1 may make the discriminator too strong
        self.parser.add_argument('--lambda_res', type=float, default=0)  # for noise generation only
        self.parser.add_argument('--UseNoise', type=int, default=0)  # for noise generation only
        self.parser.add_argument('--which_res_loss', type=str, default='C2N')  # for noise generation only

        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        #args = vars(self.opt)

        #print('------------ Options -------------')
        #for k, v in sorted(args.items()):
        #    print('%s: %s' % (str(k), str(v)))
        #print('-------------- End ----------------')
        return self.opt
