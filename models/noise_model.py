import numpy as np
import torch, random
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import cv2
import torch.nn.functional as F


def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)  # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1))  # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)


def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p - 1) / 2), ] * 4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y


class NoiseModel(BaseModel):
    def name(self):
        return 'CycleGANCtrlModel'

    def initialize(self, opt):
        """
        Works on GAN-based unsupervised noise generation is limited; otherwise we prefer to integrate MonoPix with them
        Presently, the code structure is mainly borrowed from EnlightgenGAN, like vgg, patchD...
        We did not use patchD in noise generation, however we note it is optional
        An extra color loss is used. However, it seems that this loss does not bring much difference.
        """

        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.input_B = self.Tensor(nb, opt.output_nc, size, size).to(
            torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.enhance_level_A2B = self.Tensor(nb, 1, 1, 1).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.enhance_level_B2A = self.Tensor(nb, 1, 1, 1).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))

        if opt.vgg > 0 and self.isTrain:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:  # false
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda(int(self.opt.gpu_ids[0]))
            self.vgg_loss.cuda(int(self.opt.gpu_ids[0]))  # 将VGG的推理放到GPU里面
            self.vgg = networks.load_vgg16(opt.vgg_path, self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

        # note. in original CycleGAN codes, the naming is different from those used in the paper.
        # Code (vs. MonoPix paper):
        # G_A (G_{X2Y})  G_B(G_{Y2X})
        # D_A (D_Y)   D_B (D_X)

        skip = True
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.Gnorm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)

        if self.isTrain:
            use_sigmoid = False  # we always use lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.Dnorm, use_sigmoid, self.gpu_ids)

            if self.opt.patchD:
                self.netD_PA = networks.define_D(opt.input_nc, opt.ndf,
                                                 opt.which_model_netD,
                                                 opt.n_layers_patchD, opt.Dnorm, use_sigmoid, self.gpu_ids)

        if opt.resume_ckpt:
            which_epoch = opt.which_epoch
            try:
                self.load_network(self.netG_A, 'G_A', which_epoch)
            except:
                raise Exception('No G_A ckpt found')

            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_PA, 'D_PA', which_epoch)

        if self.isTrain and not self.opt.debug:
            self.old_lr = opt.lr
            if not self.opt.IP_MONO:
                self.fake_A_pool = ImagePool(opt.pool_size)
                self.fake_B_pool = ImagePool(opt.pool_size)
                print('[*]------------use ImagePool')
            else:
                self.fake_A_pool = ImagePool_Mono(opt.pool_size)
                self.fake_B_pool = ImagePool_Mono(opt.pool_size)
                print('[*]------------use ImagePool_Mono')

            # define loss functions: GAN loss, monotonicity loss, and color loss (on the residual)
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)
            self.criterionMono = lambda x: torch.clamp(self.opt.margin - (x[1::2] - x[::2]), 0., 100.).pow(2).mean()
            self.res_crit = torch.nn.L1Loss()

            # other utils. though we did not use them
            self.gaussian_kernel = get_gausskernel(5).to(
                torch.device("cuda:{}".format(self.opt.gpu_ids[0]))).requires_grad_(False)
            self.blur_core = gaussblur

            # initialize optimizers
            if self.opt.patchD:
                self.optimizer_D_PA = torch.optim.Adam(self.netD_PA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_PA)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        enhance_level_A2B = input['enhance_level_A2B']

        # generate contrastive intensities
        # for a given c1, we generate c2 by c2=c1+uniform()*(1-c1)
        # `pseudo_best_enlvl` is always 1.0, though we once tried with other settings
        input_A = util.smaple_n_times(2, input_A)
        input_B = util.smaple_n_times(2, input_B)

        enh_level_regularize_A2B = []
        for x_ in enhance_level_A2B:
            x__ = x_ + torch.abs(x_ - self.opt.pseudo_best_enlvl) * torch.rand(1) * (
                -1 if x_ > torch.ones_like(x_) * self.opt.pseudo_best_enlvl else 1)
            enh_level_regularize_A2B.append(x__)

        enh_level_regularize_A2B = torch.stack(enh_level_regularize_A2B)
        enhance_level_A2B = torch.cat((enhance_level_A2B.unsqueeze(0), enh_level_regularize_A2B.unsqueeze(0)), dim=0)
        enhance_level_A2B = enhance_level_A2B.transpose(1, 0).contiguous().view(enhance_level_A2B.shape[1]*2, 1, 1, 1)

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.enhance_level_A2B.resize_(enhance_level_A2B.size()).copy_(enhance_level_A2B)


    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def pred_special(self, input_image_, g_atten, enh_lvl):
        if not self.opt.debug:
            with torch.no_grad():
                self.netG_A.eval()
                res, _ = self.netG_A(input_image_, None, enh_lvl)
                self.netG_A.train()
                return res.cpu().numpy()
        else:
            self.input_A.resize_(input_image_.size()).copy_(input_image_)
            self.input_B.resize_(input_image_.size()).copy_(input_image_)
            self.input_img.resize_(input_image_.size()).copy_(input_image_)
            self.enhance_level.resize_(enh_lvl.size()).copy_(enh_lvl)

    def pred_special_single(self, input_image_, g_atten, enh_lvl, onevariable=False, name='test', dir='AtoB'):

        with torch.no_grad():
            if dir == 'AtoB':
                self.netG_A.eval()
                res, lat = self.netG_A(input_image_, g_atten, enh_lvl, onevariable)
                res = res.clamp_(-1, 1).cpu().numpy()
            else:
                self.netG_B.eval()
                res, lat = self.netG_B(input_image_, g_atten, enh_lvl, onevariable)
                res = res.clamp_(-1, 1).cpu().numpy()

            from skimage import io
            for idx, res_ in enumerate(res):
                res_ = np.transpose(res_, (1, 2, 0))
                res_ = np.minimum(res_, 1)
                res_ = np.maximum(res_, -1)
                io.imsave('./visualize_demo/{}_Img_{}.png'.format(name, idx),
                          ((res_ + 1) / 2 * 255.).astype(np.uint8))

    def pred_special_test(self, input_image_, g_atten, enh_lvl, onevariable=True):
        if self.opt.debug:
            with torch.no_grad():
                vis = 0
                self.netG_A.eval()
                if self.opt.vis_IN:
                    res, _, vis = self.netG_A.forward(input_image_, None, enh_lvl)
                else:
                    res, _ = self.netG_A.forward(input_image_, None, enh_lvl)
                res = res.cpu().numpy()
                if self.opt.vis_IN:
                    return res, vis
                else:
                    return res, None

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # monotonicity loss
        if self.opt.lambda_mono>0:
            loss_monoD = self.criterionMono(pred_fake)
        else:
            loss_monoD = 0
        return loss_D, loss_monoD

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A, self.loss_monoD_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A_total = self.loss_D_A + self.loss_monoD_A*self.opt.lambda_mono
        self.loss_D_A_total.backward()

    def backward_D_P(self):
        """MonoPix does not use patch discriminator, however it is an option"""
        loss_D_P, loss_D_P_MONO = self.backward_D_basic(self.netD_PA, self.real_patch, self.fake_patch)
        # more?
        if self.opt.patchD_3 > 0:  # it is 5.
            for i in range(self.opt.patchD_3):
                loss_dp_1, loss_dp_mono = self.backward_D_basic(self.netD_PA, self.real_patch_1[i],
                                                                self.fake_patch_1[i])
                loss_D_P += loss_dp_1
                loss_D_P_MONO += loss_dp_mono
            self.loss_D_P = loss_D_P / float(self.opt.patchD_3 + 1)
            self.loss_D_P_MONO = loss_D_P_MONO / float(self.opt.patchD_3 + 1)
        else:
            self.loss_D_P = loss_D_P
            self.loss_D_P_MONO = loss_D_P_MONO
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P * 2
            self.loss_D_P_MONO = self.loss_D_P_MONO * 2

        self.loss_D_P_all = self.loss_D_P + self.loss_D_P_MONO*self.opt.lambda_mono
        self.loss_D_P_all.backward()


    def backward_G(self):

        # forward process
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, None, self.enhance_level_A2B)

        # Identity loss
        # we do not use it
        if self.opt.monoidt:
            self.fake_B_monoidt, _ = self.netG_A.forward(self.real_A[::2, :], None,
                                                             torch.zeros_like(self.enhance_level_A2B[::2, :]))

        # feeding into discriminator
        # GAN loss
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)

        # monotonicity loss
        self.loss_monoG_A2B = self.criterionMono(pred_fake) if self.opt.lambda_mono>0 else 0

        # patchD. We do not use it
        if self.opt.patchD:
            w = self.real_A.size(3)
            h = self.real_A.size(2)
            w_offset = random.randint(0, max(0, w - self.opt.patchSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.patchSize - 1))

            # generate one patch
            self.fake_patch = self.fake_B[:, :, h_offset:h_offset + self.opt.patchSize,
                              w_offset:w_offset + self.opt.patchSize]
            self.real_patch = self.real_B[:, :, h_offset:h_offset + self.opt.patchSize,
                              w_offset:w_offset + self.opt.patchSize]
            self.input_patch = self.real_A[:, :, h_offset:h_offset + self.opt.patchSize,
                               w_offset:w_offset + self.opt.patchSize]

        # multi-time patch loss. We do not use it.
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            w = self.real_A.size(3)
            h = self.real_A.size(2)

            for i in range(self.opt.patchD_3):
                w_offset_1 = random.randint(0, max(0, w - self.opt.patchSize - 1))
                h_offset_1 = random.randint(0, max(0, h - self.opt.patchSize - 1))
                self.fake_patch_1.append(self.fake_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.real_patch_1.append(self.real_B[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                         w_offset_1:w_offset_1 + self.opt.patchSize])
                self.input_patch_1.append(self.real_A[:, :, h_offset_1:h_offset_1 + self.opt.patchSize,
                                          w_offset_1:w_offset_1 + self.opt.patchSize])

        # patch gan loss. We do not use it.
        loss_G_A_patch = 0
        loss_G_A_patch_MONO = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_PA.forward(self.fake_patch)
            loss_G_A_patch += self.criterionGAN(pred_fake_patch, True)
            loss_G_A_patch_MONO += self.criterionMono(pred_fake_patch)
        if self.opt.patchD_3 > 0:
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_PA.forward(self.fake_patch_1[i])
                loss_G_A_patch += self.criterionGAN(pred_fake_patch_1, True)
                loss_G_A_patch_MONO += self.criterionMono(pred_fake_patch_1)

        if not self.opt.D_P_times2:
            self.loss_G_A_patch = loss_G_A_patch / float(self.opt.patchD_3 + 1)  # avg glob and patch
            self.loss_G_A_patch_MONO = loss_G_A_patch_MONO / float(self.opt.patchD_3 + 1)
        else:
            self.loss_G_A_patch = loss_G_A_patch / float(self.opt.patchD_3 + 1) * 2
            self.loss_G_A_patch_MONO = loss_G_A_patch_MONO / float(self.opt.patchD_3 + 1) * 2

        # color loss for noise generation
        # C2N comes from ICCV21 paper "C2N: Practical Generative Noise Modeling for Real-World Denoising"
        if self.opt.lambda_res > 0:
            if self.opt.which_res_loss == 'C2N':
                self.loss_res = self.res_crit((self.real_A - self.fake_B).mean((-1, -2)),
                                                  torch.zeros_like(self.real_A[:, :, 0, 0]))
            elif self.opt.which_res_loss == 'pixel_lvl_mean':
                self.loss_res = self.res_crit((self.real_A - self.fake_B).mean((-3)),
                                                  torch.zeros_like(self.real_A[:, 0, :]))
            elif self.opt.which_res_loss == 'l1':
                self.loss_res = self.res_crit(self.fake_B, self.real_A)
            elif self.opt.which_res_loss == 'gaussian':
                self.loss_res = self.res_crit(self.real_A, self.blur_core(self.fake_B, self.gaussian_kernel))
        else:
            self.loss_res = 0

        # We use vgg loss. It helps to stabilize the training
        if self.opt.vgg > 0:
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg, self.fake_B,
                                                             self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0
        else:
            self.loss_vgg_b = 0

        if self.opt.patch_vgg:  # we do not use it.
            loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                            self.fake_patch, self.input_patch) * self.opt.vgg

            if self.opt.patchD_3 > 0:
                for i in range(self.opt.patchD_3):
                    if not self.opt.IN_vgg:
                        loss_vgg_patch += self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                         self.fake_patch_1[i],
                                                                         self.input_patch_1[i]) * self.opt.vgg
                    else:
                        loss_vgg_patch += self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                                                                               self.fake_patch_1[i],
                                                                               self.input_patch_1[i]) * self.opt.vgg
                self.loss_vgg_b += loss_vgg_patch / float(self.opt.patchD_3 + 1)
            else:
                self.loss_vgg_b += loss_vgg_patch

        # we do not use identity loss
        if self.opt.monoidt:
            self.loss_A_monoidt = self.criterionIdt(self.fake_B_monoidt, self.real_A[::2, :])
        else:
            self.loss_A_monoidt = 0

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_A_patch + \
                      self.loss_vgg_b + self.loss_monoG_A2B * self.opt.lambda_mono + \
                      self.loss_G_A_patch_MONO * self.opt.lambda_mono + \
                      self.loss_A_monoidt * self.opt.lambda_A * 0.2 + \
                      self.loss_res * self.opt.lambda_res

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.opt.grad_clip:
            util.clip_gradient(self.optimizer_G, self.opt.grad_clip)
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        if self.opt.grad_clip:
            util.clip_gradient(self.optimizer_D_A, self.opt.grad_clip)
        self.optimizer_D_A.step()
        # D_P
        if self.opt.patchD:
            self.optimizer_D_PA.zero_grad()
            self.backward_D_P()
            if self.opt.grad_clip:
                util.clip_gradient(self.optimizer_D_PA, self.opt.grad_clip)
            self.optimizer_D_PA.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        MONO_A2B = self.loss_monoG_A2B.item()

        vgg = self.loss_vgg_b.item() if self.opt.vgg > 0 else 0
        monoidtA = self.loss_A_monoidt.item() if self.opt.monoidt else 0
        Res_A2B = self.loss_res.item() if self.opt.lambda_res > 0 else 0


        return OrderedDict([('D_A', D_A), ('G_A', G_A),
                            ("vgg", vgg),
                            ('MONO_A2B', MONO_A2B),
                            ('monoidtA', monoidtA),
                            ('Res_A2B', Res_A2B)])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
        else:
            latent_real_A = None
        real_B = util.tensor2im(self.real_B.data)


        return OrderedDict(
            [('real_A', real_A), ('real_B', real_B), ('fake_B', fake_B), ('latent_real_A', latent_real_A)])


    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        if self.opt.patchD:
            self.save_network(self.netD_PA, 'D_PA', label, self.gpu_ids)

    def update_learning_rate(self):

        lrd = self.opt.lr / self.opt.niter_decay  # 100次衰减到0
        lr = self.old_lr - lrd

        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_PA.param_groups:
                param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
