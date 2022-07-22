import os, yaml
from options.test_options import TestOptions
from models.models import create_model
import torch
from skimage import io, transform
import numpy as np


def topn1(img):
    # lpips accepts [-1, +1] as the input
    # https://github.com/richzhang/PerceptualSimilarity
    # input: [0, 255]; output:[-1, +1]
    return ((img / 255.) - 0.5) * 2


def run_predict(input_path, gt_path, save_path,
                N_search=7, lowerb=0, higherb=1, inference_type='ternary_search', ts_critic='psnr'):
    with torch.no_grad():
        if inference_type == 'exhaustive':
            for img_name in os.listdir(input_path):

                save_folder = os.path.join(save_path, img_name[:-4])  # we create for each image a folder.
                os.makedirs(save_folder, exist_ok=True)

                test_img = io.imread(os.path.join(input_path, img_name))
                if test_img.shape[0]>256 and 'LOL' not in opt.gt_path:  # we do not resize LOL images
                    test_img = transform.resize(test_img, [256, 256], anti_aliasing=True, preserve_range=True)
                io.imsave(os.path.join(save_folder, img_name[:-4]) + '_enhlvl_{:02d}.png'.format(99), test_img.astype(np.uint8)) # input
                test_img = (test_img/ 255. - 0.5) * 2
                test_img = np.transpose(test_img, (2, 0, 1))
                r, g, b = test_img[0] + 1, test_img[1] + 1, test_img[2] + 1
                A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
                test_bath = np.tile(test_img[np.newaxis], (N_search+1, 1, 1, 1))
                tets_gray = np.tile(A_gray[np.newaxis], (N_search+1, 1, 1, 1))

                # `gray` is used for LowLight enhance only.
                input_A = torch.tensor(test_bath, dtype=torch.float32, device=device)
                input_A_gray = torch.tensor(tets_gray, dtype=torch.float32, device=device)

                enhance_level = torch.linspace(lowerb, higherb, N_search+1).view(N_search+1, 1, 1, 1).to(device)
                enhanced, _ = Generator.forward(input_A, input_A_gray, enhance_level)

                for enh_idx, img_ in enumerate(enhanced.cpu().numpy()):
                    img_ = img_.transpose(1, 2, 0)
                    img_ = np.minimum(img_, 1)
                    img_ = np.maximum(img_, -1)
                    img_ = ((img_ + 1) / 2. * 255.).astype(np.uint8)
                    io.imsave(os.path.join(save_folder, img_name[:-4]) + '_enhlvl_{:02d}.png'.format(
                        int(10 * enhance_level[enh_idx])), img_)

        elif inference_type == 'ternary_search':
            assert os.path.isdir(gt_path)
            if ts_critic == 'lpips':
                import lpips
                lpips_calfun = lpips.LPIPS(net='alex').to(device)
            else:
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity

            for img_name in os.listdir(input_path):
                save_folder = os.path.join(save_path, img_name[:-4])  # remove the suffix
                os.makedirs(save_folder, exist_ok=True)

                gt_img = io.imread(os.path.join(gt_path, img_name))
                bs_cnt, lower, higher = 0, lowerb, higherb
                test_img = io.imread(os.path.join(input_path, img_name))
                if test_img.shape[0]>256 and 'LOL' not in opt.gt_path:  # we do not resize LOL images
                    test_img = transform.resize(test_img, [256, 256], anti_aliasing=True, preserve_range=True)
                io.imsave(os.path.join(save_folder, img_name[:-4]) + '_enhlvl_{:02d}.png'.format(99), test_img) # input
                test_img = (test_img/ 255. - 0.5) * 2
                test_img = np.transpose(test_img, (2, 0, 1))
                r, g, b = test_img[0] + 1, test_img[1] + 1, test_img[2] + 1
                A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.
                img_batch = torch.tensor(test_img, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(0)
                atten_batch = torch.tensor(A_gray, dtype=torch.float32, device=torch.device("cuda:0")).unsqueeze(
                    0).unsqueeze(0)

                while bs_cnt < N_search:
                    mid1 = lower + (higher - lower) / 3
                    mid2 = higher - (higher - lower) / 3
                    mid1 = torch.tensor(mid1, dtype=torch.float32, device=device).view(1, 1, 1, 1)
                    mid2 = torch.tensor(mid2, dtype=torch.float32, device=device).view(1, 1, 1, 1)

                    # we all use `skip` Unet. the second return type is residual.
                    mid1_res, _ = Generator.forward(img_batch, atten_batch, mid1)
                    mid2_res, _ = Generator.forward(img_batch, atten_batch, mid2)

                    mid1_res = mid1_res[0].detach().cpu().numpy().transpose(1, 2, 0)
                    mid1_res = np.minimum(mid1_res, 1)
                    mid1_res = np.maximum(mid1_res, -1)
                    mid1_res = ((mid1_res + 1) / 2. * 255.).astype(np.uint8)

                    mid2_res = mid2_res[0].detach().cpu().numpy().transpose(1, 2, 0)
                    mid2_res = np.minimum(mid2_res, 1)
                    mid2_res = np.maximum(mid2_res, -1)
                    mid2_res = ((mid2_res + 1) / 2. * 255.).astype(np.uint8)

                    if ts_critic == 'psnr':
                        mid1_score = peak_signal_noise_ratio(gt_img, mid1_res)
                        mid2_score = peak_signal_noise_ratio(gt_img, mid2_res)
                        if mid1_score <= mid2_score:
                            lower = mid1
                        else:
                            higher = mid2
                    elif ts_critic == 'ssim':
                        mid1_score = structural_similarity(gt_img, mid1_res, multichannel=True)
                        mid2_score = structural_similarity(gt_img, mid2_res, multichannel=True)
                        if mid1_score <= mid2_score:
                            lower = mid1
                        else:
                            higher = mid2
                    else:
                        mid1_score = lpips_calfun(  # first norm to [-1, +1], then calculate LPIPS
                            torch.FloatTensor(topn1(gt_img).transpose(2, 0, 1)).unsqueeze(0).to(device),
                            torch.FloatTensor(topn1(mid1_res).transpose(2, 0, 1)).unsqueeze(0).to(device))
                        mid2_score = lpips_calfun(
                            torch.FloatTensor(topn1(gt_img).transpose(2, 0, 1)).unsqueeze(0).to(device),
                            torch.FloatTensor(topn1(mid2_res).transpose(2, 0, 1)).unsqueeze(0).to(device))
                        if mid1_score <= mid2_score:  # negative to LPIPS
                            higher = mid2
                        else:
                            lower = mid1

                    bs_cnt += 1

                final_enhance_level = torch.tensor((mid1+mid2)/2, dtype=torch.float32, device=device).view(1, 1, 1, 1)
                final_res, _ = Generator.forward(img_batch, atten_batch, final_enhance_level)
                final_res = final_res[0].detach().cpu().numpy().transpose(1, 2, 0)
                final_res = np.minimum(final_res, 1)
                final_res = np.maximum(final_res, -1)
                final_res = ((final_res + 1) / 2. * 255.).astype(np.uint8)

                print('final enhance level for ', img_name, 'is ', final_enhance_level)
                io.imsave(os.path.join(save_folder, img_name[:-4]) + '_enhlvl_{:02d}.png'.format(
                    int(10 * final_enhance_level[0].item())), final_res)

if __name__=='__main__':

    opt = TestOptions().parse()
    os.makedirs(opt.save_path, exist_ok=True)
    print('loading from config')

    if opt.config:
        f = yaml.safe_load(open(opt.config, 'r'))
        for kk, vv in f.items():
            setattr(opt, kk, vv)

    print('--------args----------')
    for k in list(sorted(vars(opt).keys())):
        print('%s: %s' % (k, vars(opt)[k]))
    print('--------args----------\n')

    model = create_model(opt)

    if opt.which_direction != 'AtoB':
        assert model.netG_B

    Generator = model.netG_B if opt.which_direction != 'AtoB' else model.netG_A
    Generator.eval()


    device = 'cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else 'cpu:0'
    device = torch.device(device)

    # run prediction
    save_folder = os.path.join(opt.save_path, opt.name + '_{}_{}'.format(opt.tag, opt.inf_type))
    os.makedirs(save_folder, exist_ok=True)
    run_predict(opt.input_path, opt.gt_path, save_folder, N_search=opt.N_search,
                lowerb=opt.lowerb, higherb=opt.higherb, inference_type=opt.inf_type, ts_critic=opt.ts_critic)

