import time, os, sys, yaml, random, torch
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from skimage import io
from util.MonoPix_utils import get_special_input
import numpy as np

opt = TrainOptions().parse()

# merge yaml into config
if opt.config:
    print('loading from config')
    f = yaml.safe_load(open(opt.config, 'r'))
    for kk, vv in f.items():
        setattr(opt, kk, vv)

print('--------args----------')
for k in list(sorted(vars(opt).keys())):
    print('%s: %s' % (k, vars(opt)[k]))
print('--------args----------\n')

# seeds
def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

init_seeds(opt.seeds)
os.makedirs('./visualize/'+opt.name, exist_ok=True)
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
os.makedirs(expr_dir, exist_ok=True)

# save to the disk

file_name = os.path.join(expr_dir, 'opt.txt')
with open(file_name, 'wt') as opt_file:
    opt_file.write('------------ Options -------------\n')
    for k in list(sorted(vars(opt).keys())):
        opt_file.write(('%s: %s \n' % (k, vars(opt)[k])))
    opt_file.write('-------------- End ----------------\n')


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
total_steps = 0


for epoch in range(1, opt.niter + opt.niter_decay + 1):  # 201
    INNER_ITER = 0
    for i, data in enumerate(dataset):
        INNER_ITER += 1
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()
        
        if INNER_ITER % opt.display_freq == 0:
            os.makedirs('./visualize/'+opt.name+'/epoch{}'.format(epoch), exist_ok=True)
            vis = model.get_current_visuals()
            for item in ['real_A', 'real_B', 'fake_B']:
                io.imsave('./visualize/' + opt.name + '/epoch{}/{}_{}.png'.format(epoch, item, INNER_ITER), vis[item])
            for item in ['fake_A', 'latent_real_A', 'rec_B', 'rec_A']:
                if item in vis:
                    io.imsave('./visualize/'+opt.name+'/epoch{}/{}_{}.png'.format(epoch, item, INNER_ITER), vis[item])
            if opt.self_attention:
                io.imsave('./visualize/'+opt.name+'/epoch{}/attenA_{}.png'.format(epoch, INNER_ITER), vis['self_attention'])

            spe_input = get_special_input(opt.name, torch.device("cuda:{}".format(opt.gpu_ids[0])))
            res = model.pred_special(*spe_input)
            for idx, res_ in enumerate(res):
                res_ = np.transpose(res_, (1,2,0))
                res_ = np.minimum(res_, 1)
                res_ = np.maximum(res_, -1)
                io.imsave('./visualize/'+opt.name+'/epoch{}/INTER_ITER{}_{}.png'.format(epoch, INNER_ITER, idx), ((res_+1)/2*255.).astype(np.uint8))

        if INNER_ITER % opt.print_freq == 0:
            errors = model.get_current_errors()
            print('epoch [{}/200]  loss: '.format(epoch) + '  '.join('{}={:.3f}'.format(kk, vv) for kk, vv in errors.items()))
            sys.stdout.flush()

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(epoch)

    if epoch > opt.niter or opt.custom_lr:
        model.update_learning_rate()
