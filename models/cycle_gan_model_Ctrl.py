import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool, ImagePool_Mono
from .base_model import BaseModel
from . import networks
import sys


class CycleGANCtrlModel(BaseModel):
    def name(self):
        return 'CycleGANCtrlModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.input_B = self.Tensor(nb, opt.output_nc, size, size).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.enhance_level_A2B = self.Tensor(nb, 1, 1, 1).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))
        self.enhance_level_B2A = self.Tensor(nb, 1, 1, 1).to(torch.device('cuda:{}'.format(self.opt.gpu_ids[0])))

        # note. in original CycleGAN codes, the naming is different from those used in the paper.
        # Code (vs. MonoPix paper):
        # G_A (G_{X2Y})  G_B(G_{Y2X})
        # D_A (D_Y)   D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.Gnorm, None, self.gpu_ids, skip=self.opt.skip, opt=opt)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.Gnorm, None, self.gpu_ids, skip=self.opt.skip, opt=opt)

        # define networks
        if self.isTrain:
            use_sigmoid = False  # we always use lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.Dnorm, use_sigmoid, self.gpu_ids)
    
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.Dnorm, use_sigmoid, self.gpu_ids)

        # if resume checkpoints
        if opt.resume_ckpt:
            try: self.load_network(self.netG_A, 'G_A', opt.which_epoch)
            except: raise Exception('No G_A ckpt found')
            
            try: self.load_network(self.netG_B, 'G_B', opt.which_epoch)
            except:  raise Exception('No G_B ckpt found')
                
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', opt.which_epoch)
                self.load_network(self.netD_B, 'D_B', opt.which_epoch)

        # define loss functions and utils.
        if self.isTrain:
            self.old_lr = opt.lr

            # Note. this following codes on ImagePool may be confusing for some readers.
            # in original CycleGAN paper, image pool is used to make G and D irrelevant to the same training batch
            # In MonoPix, we tried to modify ImagePool to make two contrastive sample come together (ImagePool_Mono)
            # However, sometimes using `ImagePool_Mono` makes discriminator too strong and increase the instability
            # So we keep using the original ImagePool, considering the following reasons:
            # On one hand, it makes training G and D irrelevant,
            # on the other hand, it confuses the discriminator when calculating monotonicity loss
            # In EnligtenGAN, the authors do not use image pool. We follow their settings on LowLight enhancement
            if not self.opt.IP_MONO:
                self.fake_A_pool = ImagePool(opt.pool_size)
                self.fake_B_pool = ImagePool(opt.pool_size)
                print('[*]------------use ImagePool')
            else:
                self.fake_A_pool = ImagePool_Mono(opt.pool_size)
                self.fake_B_pool = ImagePool_Mono(opt.pool_size)
                print('[*]------------use ImagePool_Mono')

            # define loss functions. `Df` denotes domain fidelity
            self.criterionGAN = networks.GANLoss(use_lsgan=True, tensor=self.Tensor)  # we use lsgan
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMono = lambda x: torch.clamp(self.opt.margin - (x[1::2] - x[::2]), 0., 100.).pow(2).mean()
            self.criterionMono_Df = lambda x: torch.clamp(self.opt.margin - (x[::2] - x[1::2]), 0., 100.).pow(2).mean()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            self.netG_A.train()
            self.netG_B.train()
            self.netD_A.train()
            self.netD_B.train()
        else:
            self.netG_A.eval()
            self.netG_B.eval()
        print('-----------------------------------------------')


    def set_input(self, input):
        # input is a dictionary
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        enhance_level_A2B = input['enhance_level_A2B']
        enhance_level_B2A = input['enhance_level_B2A']

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


        # if using the same $c$ for A2B and B2A. default is False
        if self.opt.use_samelvl:
            enhance_level_B2A = -enhance_level_A2B
        else:
            enh_level_regularize_B2A = []
            for x_ in enhance_level_B2A:
                x__ = x_ + torch.abs(x_ - self.opt.pseudo_best_enlvl) * torch.rand(1) * (
                -1 if x_ > torch.ones_like(x_) * self.opt.pseudo_best_enlvl else 1)
                enh_level_regularize_B2A.append(x__)

            enh_level_regularize_B2A = torch.stack(enh_level_regularize_B2A)
            enhance_level_B2A = torch.cat((enhance_level_B2A.unsqueeze(0), enh_level_regularize_B2A.unsqueeze(0)), dim=0)
            enhance_level_B2A = enhance_level_B2A.transpose(1, 0).contiguous().view(enhance_level_B2A.shape[1]*2, 1, 1, 1)


        # the following codes set for input tensor a proper device. They are tedious, and not a must (from EnlightenGAN)
        # may be deprecated in future
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.enhance_level_A2B.resize_(enhance_level_A2B.size()).copy_(enhance_level_A2B)
        self.enhance_level_B2A.resize_(enhance_level_B2A.size()).copy_(enhance_level_B2A)


    def forward(self):
        # may be deprecated
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

                
    def backward_D_basic(self, netD, real, fake, domain_fidelity=False):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if not domain_fidelity:
            loss_monoD = self.criterionMono(pred_fake)
        else:
            loss_monoD = self.criterionMono_Df(pred_fake)
        return loss_D, loss_monoD

    def backward_D_A(self):
        # using a different set of input image samples to decouple training.
        self.fake_B = self.fake_B_pool.query(self.fake_B)
        self.fake_A = self.fake_A_pool.query(self.fake_A)

        # GAN loss and Monotonicity loss: D_A(fakeB)
        self.loss_D_A, self.loss_monoD_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_A_total = self.loss_D_A + self.loss_monoD_A*self.opt.lambda_mono

        # Domain fidelity: D_A(fakeA)
        # those with low translation intensity better preserve the source-domain feature
        if self.opt.cal_D_Df:
            _, self.loss_monoD_A_Df = self.backward_D_basic(self.netD_A, self.real_B, self.fake_A, domain_fidelity=True)
        else:
            self.loss_monoD_A_Df = 0
        self.loss_D_A_total += self.loss_monoD_A_Df*self.opt.lambda_df
        self.loss_D_A_total.backward()

    def backward_D_B(self):
        # we have already sampled from fake pool
        # GAN loss and Monotonicity loss: D_B(fakeA)
        self.loss_D_B, self.loss_monoD_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        self.loss_D_B_total = self.loss_D_B + self.loss_monoD_B*self.opt.lambda_mono

        # Domain fidelity: D_B(fakeB)
        if self.opt.cal_D_Df:
            _, self.loss_monoD_B_Df = self.backward_D_basic(self.netD_B, self.real_A, self.fake_B, domain_fidelity=True)
        else:
            self.loss_monoD_B_Df = 0
        self.loss_D_B_total += self.loss_monoD_B_Df*self.opt.lambda_df

        self.loss_D_B_total.backward()

    def backward_G(self):

        # Stage 1: A2B
        # realA -> G_A -> fakeB
        # D_A(fakeB): GAN loss and monotonicity loss
        # D_B(fakeB): domain fidelity loss

        # forward process
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, None, self.enhance_level_A2B)

        # identity loss. we do not use it
        if self.opt.monoidt:
            self.fake_B_monoidt, _ = self.netG_A.forward(self.real_A[::2,:], None, torch.zeros_like(self.enhance_level_A2B[::2,:]))

        # feeding into discriminator
        # GAN loss
        pred_fake = self.netD_A.forward(self.fake_B)
        self.loss_G_A = self.criterionGAN(pred_fake, True)

        # monotonicity loss
        self.loss_monoG_A2B = self.criterionMono(pred_fake) if self.opt.lambda_mono>0 else 0

        # Domain fidelity loss
        if self.opt.cal_G_Df > 0:
            pred_fake_Df = self.netD_B.forward(self.fake_B)
            self.loss_monoG_A2B_Df = self.criterionMono_Df(pred_fake_Df)
        else:
            self.loss_monoG_A2B_Df = 0

        # Stage 2: B2A
        # fakeA <- G_B <- realB
        # D_B(fakeA): GAN loss and monotonicity loss
        # D_A(fakeA): domain fidelity  loss

        # forward process
        self.fake_A, self.latent_real_B = self.netG_B.forward(self.real_B, None, self.enhance_level_B2A)

        # identity loss. we do not use it
        if self.opt.monoidt:
            self.fake_A_monoidt, _ = self.netG_B.forward(self.real_B[::2,:], None, torch.zeros_like(self.enhance_level_B2A[::2,:]))

        # feeding into discriminator
        # GAN loss
        pred_fake = self.netD_B.forward(self.fake_A)
        self.loss_G_B = self.criterionGAN(pred_fake, True)
            
        # monotonicity loss
        self.loss_monoG_B2A = self.criterionMono(pred_fake) if self.opt.lambda_mono>0 else 0

        # Domain fidelity loss
        if self.opt.cal_G_Df > 0:
            pred_fake_Df = self.netD_A.forward(self.fake_A)
            self.loss_monoG_B2A_Df = self.criterionMono_Df(pred_fake_Df)
        else:
            self.loss_monoG_B2A_Df = 0

        # Stage 3: cycle consistency
        # Forward cycle loss: fakeB -> G_B -> realA
        if self.opt.lambda_A > 0:
            # reconstruct with the same intensity
            self.rec_A, self.latent_fake_B = self.netG_B.forward(self.fake_B, None, self.enhance_level_A2B)
            self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_A
        else:
            self.loss_cycle_A = 0

        # Backward cycle loss: fakeA -> G_A -> realB
        if self.opt.lambda_B > 0:
            self.rec_B, self.latent_fake_A = self.netG_A.forward(self.fake_A, None, self.enhance_level_B2A)
            self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_B
        else:
            self.loss_cycle_B = 0

        # identity loss. we do not use it
        if self.opt.monoidt:
            self.loss_A_monoidt = self.criterionIdt(self.fake_B_monoidt, self.real_A[::2,:])
            self.loss_B_monoidt = self.criterionIdt(self.fake_A_monoidt, self.real_B[::2,:])
        else:
            self.loss_A_monoidt = 0
            self.loss_B_monoidt = 0

        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + \
                      self.loss_monoG_A2B*self.opt.lambda_mono + self.loss_monoG_B2A*self.opt.lambda_mono + \
                      self.loss_A_monoidt*self.opt.lambda_A*0.2+ self.loss_B_monoidt*self.opt.lambda_B*0.2 + \
                      self.loss_monoG_A2B_Df * self.opt.lambda_df + self.loss_monoG_B2A_Df * self.opt.lambda_df

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        # self.fake has been updated.
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


    def pred_special(self, input_image_, g_atten, enh_lvl):

        with torch.no_grad():
            self.netG_A.eval()
            if self.opt.skip ==1:
                res, _ = self.netG_A(input_image_, None, enh_lvl)
            else:
                res = self.netG_A(input_image_, None, enh_lvl)
            self.netG_A.train()
            return res.cpu().numpy()


    def pred_special_test(self, input_image_, g_atten, enh_lvl, onevariable=True, dir='AtoB'):
        with torch.no_grad():
            vis = 0
            self.netG_A.eval()
            self.netG_B.eval()
            if dir == 'AtoB': model = self.netG_A
            else: model = self.netG_B

            if self.opt.vis_IN:
                res, _, vis = model.forward(input_image_, None, enh_lvl)
            else:
                res, _ = model.forward(input_image_, None, enh_lvl)
            res = res.cpu().numpy()

            return res, vis if self.opt.vis_IN else None


    def pred_special_single(self, input_image_, g_atten, enh_lvl, onevariable=False, name='test', dir='AtoB'):

        with torch.no_grad():
            if dir == 'AtoB':
                self.netG_A.eval()
                res, _ = self.netG_A(input_image_, g_atten, enh_lvl, onevariable)
                res = res.clamp_(-1, 1).cpu().numpy()
            else:
                self.netG_B.eval()
                res, _ = self.netG_B(input_image_, g_atten, enh_lvl, onevariable)
                res = res.clamp_(-1, 1).cpu().numpy()

            from skimage import io
            for idx, res_ in enumerate(res):
                res_ = np.transpose(res_, (1, 2, 0))
                res_ = np.minimum(res_, 1)
                res_ = np.maximum(res_, -1)
                io.imsave('./visualize_demo/{}_Img_{}.png'.format(name, idx),
                          ((res_ + 1) / 2 * 255.).astype(np.uint8))

    def get_current_errors(self):
        D_A = self.loss_D_A.item()
        G_A = self.loss_G_A.item()
        Cyc_A = self.loss_cycle_A.item()
        D_B = self.loss_D_B.item()
        G_B = self.loss_G_B.item()
        Cyc_B = self.loss_cycle_B.item()
        monoidtA = self.loss_A_monoidt.item() if self.opt.monoidt else 0
        monoidtB = self.loss_B_monoidt.item() if self.opt.monoidt else 0

        MONO_GA2B = self.loss_monoG_A2B.item()
        MONO_GB2A = self.loss_monoG_B2A.item()
        MONO_DA2B = self.loss_monoD_A.item()
        MONO_DB2A = self.loss_monoD_B.item()
        MONO_GA2B_DF = self.loss_monoG_A2B_Df.item() if self.opt.cal_G_Df > 0 else 0
        MONO_GB2A_DF = self.loss_monoG_B2A_Df.item() if self.opt.cal_G_Df > 0 else 0
        MONO_DA_Df = self.loss_monoD_A_Df.item() if self.opt.cal_D_Df else 0
        MONO_DB_Df = self.loss_monoD_B_Df.item() if self.opt.cal_D_Df else 0

        return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                            ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),
                            ('monoidtA', monoidtA), ('monoidtB', monoidtB),
                            ('MONO_GA2B', MONO_GA2B), ('MONO_GB2A', MONO_GB2A),
                            ('MONO_DA2B', MONO_DA2B), ('MONO_DB2A', MONO_DB2A),
                            ('MONO_GA2B_DF', MONO_GA2B_DF), ('MONO_GB2A_DF', MONO_GB2A_DF),
                            ('MONO_DA_Df',MONO_DA_Df), ('MONO_DB_Df', MONO_DB_Df)])


    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        latent_real_A = util.tensor2im(self.latent_real_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)

        rec_A = util.tensor2im(self.rec_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        latent_fake_A = util.tensor2im(self.latent_fake_A.data)

        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('latent_real_A', latent_real_A), ('rec_A', rec_A),
                            ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('latent_fake_A', latent_fake_A)])


    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        if self.opt.saveD:
            self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
            self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

    def update_learning_rate(self):

        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
    
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
