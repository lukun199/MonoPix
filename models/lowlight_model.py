import numpy as np
import torch
from torch import nn
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
import random
from . import networks
import sys


class LowLightModel(BaseModel):
    def name(self):
        return 'LowLightModel'

    def initialize(self, opt):
        """
        Note. EnlightenGAN is a unidirectional translator, form dark to normal.
        It has only one generator and one discriminator.
        MonoPix adds monotonicity loss on EnlightenGAN. Domain fidelity loss is not used.
        EnlightenGAN does not use ImagePool. MonoPix just follows the settings.
        I did not modify much on EnlightenGAN's original codes, though these codes can be significantly formatted.
        """
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        # self.input_B_OverExp = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)
        self.enhance_level = self.Tensor(nb, 1, 1, 1)

        if opt.vgg > 0 and self.isTrain:
            self.vgg_loss = networks.PerceptualLoss(opt)
            if self.opt.IN_vgg:
                self.vgg_patch_loss = networks.PerceptualLoss(opt)
                self.vgg_patch_loss.cuda()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16(opt.vgg_path, self.gpu_ids)
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif opt.fcn > 0:  # False
            self.fcn_loss = networks.SemanticLoss(opt)
            self.fcn_loss.cuda()
            self.fcn = networks.load_fcn(opt.vgg_path)
            self.fcn.eval()
            for param in self.fcn.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,  # 3, 3
                                        opt.ngf, opt.which_model_netG, opt.Gnorm, not opt.no_dropout, self.gpu_ids,
                                        skip=skip, opt=opt)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
        #                                 opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=False, opt=opt)

        if self.isTrain:
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.Dnorm, use_sigmoid, self.gpu_ids, False)  # NotPatch
            if self.opt.patchD:
                # norm: instance
                # do not use sigmoid
                self.netD_P = networks.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_patchD, opt.Dnorm, use_sigmoid, self.gpu_ids, True)

        if opt.resume_ckpt:  # resume for training
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                if self.opt.patchD:
                    self.load_network(self.netD_P, 'D_P', which_epoch)
            print('[**]------------------resumed from ckpt: ', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)  # however it is not used.
            # define loss functions MonoPix use LSGAN
            if opt.use_wgan:  # False
                self.criterionGAN = networks.DiscLossWGANGP()
            else:  # True, we use lsgan
                self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)

            self.criterionCycle = torch.nn.L1Loss()
            # self.criterionL1 = torch.nn.L1Loss()
            # self.criterionIdt = torch.nn.L1Loss()

            # MonoPix add one extra loss function
            self.criterionMono = lambda x: torch.clamp(self.opt.margin - (x[1::2] - x[::2]), 0., 100.).pow(2).mean()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            if self.opt.patchD:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        # networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            if self.opt.patchD:
                networks.print_network(self.netD_P)
            # networks.print_network(self.netD_B)
        if opt.isTrain:
            self.netG_A.train()
            # self.netG_B.train()
        else:
            self.netG_A.eval()
            # self.netG_B.eval()
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_img = input['input_img']
        input_A_gray = input['A_gray']
        enhance_level = input['enhance_level']

        # MonoPix assign two intensities for one patch
        input_A = util.smaple_n_times(2, input_A)
        input_B = util.smaple_n_times(2, input_B)
        input_img = util.smaple_n_times(2, input_img)
        input_A_gray = util.smaple_n_times(2, input_A_gray)

        # generate contrastive intensities
        # for a given c1, MonoPix generates c2 by c2=c1+uniform()*(1-c1)
        # `pseudo_best_enlvl` is always 1.0, though we once tried with other settings
        enh_level_regularize = []
        for x_ in enhance_level:
            x__ = x_ + torch.abs(x_ - self.opt.pseudo_best_enlvl) * torch.rand(1) * (
                -1 if x_ > torch.ones_like(x_) * self.opt.pseudo_best_enlvl else 1)
            enh_level_regularize.append(x__)

        enh_level_regularize = torch.stack(enh_level_regularize)
        enhance_level = torch.cat((enhance_level.unsqueeze(0), enh_level_regularize.unsqueeze(0)), dim=0).transpose(
            1,0).contiguous().view(enhance_level.shape[0] * 2, 1, 1, 1)

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.enhance_level.resize_(enhance_level.size()).copy_(enhance_level)

    # added by MonoPix
    def pred_special_single(self, input_image_, g_atten, enh_lvl, onevariable=False, name='test'):
        with torch.no_grad():
            self.netG_A.eval()
            res, _ = self.netG_A(input_image_, g_atten, enh_lvl, onevariable)
            res = res.clamp_(-1,1).cpu().numpy()
            from skimage import io
            for idx, res_ in enumerate(res):
                res_ = np.transpose(res_, (1, 2, 0))
                res_ = np.minimum(res_, 1)
                res_ = np.maximum(res_, -1)
                io.imsave('./visualize_demo/{}_Img_{}.png'.format(name, idx),
                          ((res_ + 1) / 2 * 255.).astype(np.uint8))

    # MonoPix did not use this function.
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_A_gray = Variable(self.input_A_gray, volatile=True)
        if self.opt.noise > 0:
            self.noise = Variable(torch.cuda.FloatTensor(self.real_A.size()).normal_(mean=0, std=self.opt.noise / 255.))
            self.real_A = self.real_A + self.noise
        if self.opt.input_linear:  # False
            self.real_A = (self.real_A - torch.min(self.real_A)) / (torch.max(self.real_A) - torch.min(self.real_A))
        # print(np.transpose(self.real_A.data[0].cpu().float().numpy(),(1,2,0))[:2][:2][:])
        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray)  # 这个gray到底是什么啊
        else:
            self.fake_B = self.netG_A.forward(self.real_A, self.real_A_gray)
        # self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)

    def pred_special(self, input_image_, g_atten, enh_lvl):
        if not self.opt.debug:
            with torch.no_grad():
                self.netG_A.eval()
                res, _ = self.netG_A(input_image_, g_atten, enh_lvl)
                self.netG_A.train()
                return res.cpu().numpy()


    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_D_basic(self, netD, real, fake):
        """MonoPix adds monotonicity loss"""

        pred_real = netD.forward(real)
        pred_fake = netD.forward(fake.detach())

        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # monotonicity loss
        if self.opt.lambda_mono>0:
            loss_monoD_A = self.criterionMono(pred_fake)
        else:
            loss_monoD_A = 0

        return loss_D, loss_monoD_A

    def backward_D_A(self):
        # fake_B = self.fake_B_pool.query(self.fake_B)
        fake_B = self.fake_B  # a little weird. In the original EnlightenGAN, image pool is not used.
        self.loss_D_A, self.loss_monoD_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A_total = self.loss_D_A + self.opt.lambda_mono * self.loss_monoD_A
        self.loss_D_A_total.backward()

    def backward_D_P(self):
        # first we get one.
        loss_D_P, loss_monoD_PA = self.backward_D_basic(self.netD_P, self.real_patch, self.fake_patch)
        # more patch
        if self.opt.patchD_3 > 0:  # it is 5.
            for i in range(self.opt.patchD_3):
                loss1, loss2 = self.backward_D_basic(self.netD_P, self.real_patch_1[i], self.fake_patch_1[i])
                loss_D_P += loss1
                loss_monoD_PA += loss2
            self.loss_D_P = loss_D_P / float(self.opt.patchD_3 + 1)
            self.loss_monoD_PA = loss_monoD_PA / float(self.opt.patchD_3 + 1)
        else:
            self.loss_D_P = loss_D_P
        if self.opt.D_P_times2:
            self.loss_D_P = self.loss_D_P * 2

        self.loss_D_P_total = self.loss_D_P + self.opt.lambda_mono * self.loss_monoD_PA
        self.loss_D_P_total.backward()

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)

        if self.opt.skip == 1:
            self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_img, self.real_A_gray, self.enhance_level)
        else:
            self.fake_B = self.netG_A.forward(self.real_img, self.real_A_gray)

        # patch generation and loss
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


        # multi-time patch loss
        if self.opt.patchD_3 > 0:
            self.fake_patch_1 = []
            self.real_patch_1 = []
            self.input_patch_1 = []
            self.latentPatch_1 = []
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


            # w_offset_2 = random.randint(0, max(0, w - self.opt.patchSize - 1))
            # h_offset_2 = random.randint(0, max(0, h - self.opt.patchSize - 1))
            # self.fake_patch_2 = self.fake_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.real_patch_2 = self.real_B[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]
            # self.input_patch_2 = self.real_A[:,:, h_offset_2:h_offset_2 + self.opt.patchSize,
            #        w_offset_2:w_offset_2 + self.opt.patchSize]


    def backward_G(self):
        """MonoPix adds monotonicity loss"""

        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        self.loss_monoG_A = self.criterionMono(pred_fake) if self.opt.lambda_mono>0 else 0

        # patch Discriminator.
        loss_G_A = 0
        loss_mono_A = 0
        if self.opt.patchD:
            pred_fake_patch = self.netD_P.forward(self.fake_patch)
            loss_mono_A += self.criterionMono(pred_fake_patch) if self.opt.lambda_mono>0 else 0
            loss_G_A += self.criterionGAN(pred_fake_patch, True)

        if self.opt.patchD_3 > 0:
            for i in range(self.opt.patchD_3):
                pred_fake_patch_1 = self.netD_P.forward(self.fake_patch_1[i])
                loss_mono_A += self.criterionMono(pred_fake_patch_1) if self.opt.lambda_mono>0 else 0
                loss_G_A += self.criterionGAN(pred_fake_patch_1, True)

            if not self.opt.D_P_times2:  # In original EnlightenGAN
                self.loss_G_A += loss_G_A / float(self.opt.patchD_3 + 1)  # avg glob and patch
                self.loss_monoG_A += loss_mono_A / float(self.opt.patchD_3 + 1)
            else:
                self.loss_G_A += loss_G_A / float(self.opt.patchD_3 + 1) * 2
        else:
            if not self.opt.D_P_times2:
                self.loss_G_A += loss_G_A
            else:
                self.loss_G_A += loss_G_A * 2

        # vgg loss.
        vgg_w = 1

        if self.opt.vgg > 0:  # True
            self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                             self.fake_B,
                                                             self.real_A) * self.opt.vgg if self.opt.vgg > 0 else 0  # 和母体近似。。额
            if self.opt.patch_vgg:  # True
                if not self.opt.IN_vgg:
                    loss_vgg_patch = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                    self.fake_patch, self.input_patch) * self.opt.vgg
                else:
                    loss_vgg_patch = self.vgg_patch_loss.compute_vgg_loss(self.vgg,
                                                                          self.fake_patch,
                                                                          self.input_patch) * self.opt.vgg

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
            self.loss_G = self.loss_G_A + self.loss_vgg_b * vgg_w

        else:
            self.loss_G = self.loss_G_A
            self.loss_vgg_b = 0

        self.loss_G += self.opt.lambda_mono * self.loss_monoG_A
        # self.opt.info_weight * (self.reg_A_loss + self.reg_PA_loss)

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()  # release the graph
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
            self.optimizer_D_P.zero_grad()
            self.backward_D_P()
            if self.opt.grad_clip:
                util.clip_gradient(self.optimizer_D_P, self.opt.grad_clip)
            self.optimizer_D_P.step()


    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        D_P = self.loss_D_P.item() if self.opt.patchD else 0
        G_A = self.loss_G_A.item()
        mono_G = self.loss_monoG_A.item() if self.opt.lambda_mono>0 else 0
        monoDA = self.loss_monoD_A.item() if self.opt.lambda_mono>0 else 0
        monoDP = self.loss_monoD_PA.item() if self.opt.lambda_mono>0 and self.opt.patchD else 0

        vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
        return OrderedDict([('D_A', D_A), ('G_A', G_A), ("vgg", vgg), ("D_P", D_P), ("mono_G", mono_G),
                            ('monoDA', monoDA), ("monoDP", monoDP)])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        if self.opt.skip > 0:
            latent_real_A = util.tensor2im(self.latent_real_A.data)
            latent_show = util.latent2im(self.latent_real_A.data)
            if self.opt.patchD:
                fake_patch = util.tensor2im(self.fake_patch.data)
                real_patch = util.tensor2im(self.real_patch.data)
                if self.opt.patch_vgg:
                    input_patch = util.tensor2im(self.input_patch.data)
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('input_patch', input_patch)])
                    else:  # 返回这个
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('input_patch', input_patch),
                                            ('self_attention', self_attention)])
                else:
                    if not self.opt.self_attention:
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch)])
                    else:
                        self_attention = util.atten2im(self.real_A_gray.data)
                        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                            ('latent_show', latent_show), ('real_B', real_B),
                                            ('real_patch', real_patch),
                                            ('fake_patch', fake_patch), ('self_attention', self_attention)])
            else:
                if not self.opt.self_attention:
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A),
                                        ('latent_show', latent_show), ('real_B', real_B)])
                else:
                    self_attention = util.atten2im(self.real_A_gray.data)
                    return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                        ('latent_real_A', latent_real_A), ('latent_show', latent_show),
                                        ('self_attention', self_attention)])
        else:
            if not self.opt.self_attention:
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
            else:
                self_attention = util.atten2im(self.real_A_gray.data)
                return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
                                    ('self_attention', self_attention)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        if self.opt.saveD:
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
            if self.opt.patchD:
                self.save_network(self.netD_P, 'D_P', label, self.gpu_ids)

    def update_learning_rate(self, epoch=0):

        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        if self.opt.patchD:
            for param_group in self.optimizer_D_P.param_groups:
                param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
