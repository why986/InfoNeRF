import configargparse
import numpy as np
import os
import jittor as jt
from load_blender import load_blender_data
from tqdm import trange, tqdm
import imageio
from loss import EntropyLoss
from utils import calc_psnr, calc_mse, to_byte, get_pos_encoding, batchify
from NeRF import NeRF
from render import render, render_path, get_rays

def run_network(x, d, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    x_flat = jt.reshape(x, [-1, x.shape[-1]])
    embedded = embed_fn(x_flat)

    if d is not None:
        d = d[:,None].expand(x.shape)
        d_flat = jt.reshape(d, [-1, d.shape[-1]])
        embedded_dirs = embeddirs_fn(d_flat)
        embedded = jt.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = jt.reshape(outputs_flat, list(x.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    embed_fn, input_x_channels = get_pos_encoding(10)
    embeddirs_fn = None
    if args.use_viewdirs:
            embeddirs_fn, input_d_channels = get_pos_encoding(4)
            print("input_x_channels: ", input_x_channels, "input_d_channels:", input_d_channels)
    output_channels = 5 if args.N_importance > 0 else 4
    model = NeRF(input_x_channels=input_x_channels, output_channels=output_channels, 
                    input_d_channels=input_d_channels, use_viewdirs=args.use_viewdirs)
    grad_vars = list(model.parameters())


    model_fine = None
    if args.N_importance > 0:
            model_fine = NeRF(input_x_channels=input_x_channels, output_channels=output_channels,
                                input_d_channels=input_d_channels, use_viewdirs=args.use_viewdirs)
            grad_vars += list(model_fine.parameters())

    network_query_fn = lambda x, d, model : run_network(x, d, model, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn)

    # Create optimizer
    optimizer = jt.optim.Adam(params=grad_vars, lr=args.lr, betas=(0.9, 0.999))

    global_step = 0
    # Load checkpoints
    if args.resume_ckpt is not None:
            checkpoint = jt.load(args.resume_ckpt)

            global_step = checkpoint['global_step']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.load_state_dict(checkpoint['network_fn_state_dict'])
            if model_fine is not None:
                    model_fine.load_state_dict(checkpoint['network_fine_state_dict'])

    render_kwargs_train = {
            'network_query_fn' : network_query_fn,
            'perturb' : 1.,
            'N_importance' : args.N_importance,
            'network_fine' : model_fine,
            'N_samples' : args.N_samples,
            'network_fn' : model,
            'use_viewdirs' : args.use_viewdirs,
            'white_bkgd' : args.white_bkgd,
            'raw_noise_std' : 0.,
            'entropy_ray_zvals' : args.entropy,
            'ndc' : False,
            'lindisp' : False
    }

    render_kwargs_test = render_kwargs_train.copy()
    render_kwargs_test['perturb'] = False

    return render_kwargs_train, render_kwargs_test, global_step, grad_vars, optimizer



def parse_config():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--resume_ckpt", type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument("--expname", type=str, default='nerf', help='experiment name')
    parser.add_argument("--basedir", type=str, default='.', help='base directory for experiment')
    parser.add_argument("--datadir", type=str, default='.', help='base directory for data')
    parser.add_argument("--dataset_type", type=str, default='blender', help='dataset type')
    parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                    help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--use_viewdirs", action='store_true', 
                    help='use full 5D input instead of 3D')
    parser.add_argument("--white_bkgd", action='store_true', 
                    help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lrate_decay", type=int, default=250, 
                    help='exponential learning rate decay (in 1000 steps)')

    parser.add_argument("--N_samples", type=int, default=64, 
                    help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                    help='number of additional fine samples per ray')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                    help='batch size (number of random rays per gradient step)')
    parser.add_argument("--N_iters", type=int, default=200000, 
                    help='number of iters')

    parser.add_argument("--i_video",   type=int, default=10000, 
                    help='frequency of render_poses video saving')
    parser.add_argument("--i_testset", type=int, default=10000, 
                    help='frequency of testset saving')

    parser.add_argument("--entropy", action='store_true',
                    help='using entropy ray loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=1,
                    help='entropy lambda for ray zvals entropy loss')
    parser.add_argument("--N_entropy", type=int, default=100,
                    help='number of entropy ray')

    parser.add_argument("--precrop_iters", type=int, default=0,
                    help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                    default=.5, help='fraction of img taken for central crops') 

    parser.add_argument("--fewshot", type=int, default=0, 
            help='if 0 not using fewshot, else: using fewshot')
    parser.add_argument("--train_scene", nargs='+', type=int,
                    help='id of scenes used to train')
    parser.add_argument("--render_only", action='store_true')
    parser.add_argument("--render_test", action='store_true')
    args = parser.parse_args()
    return args

def train(args):
    images, poses, render_poses, hwf, index_split = load_blender_data(args.datadir)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    print(hwf)
    H, W, focal = hwf
    train_index, val_index, test_index = index_split

    if args.fewshot > 0:
        if args.train_scene is None:
            np.random.seed(666)
            train_index = np.random.choice(train_index, args.fewshot, replace=False)
        else:
            train_index = np.array(args.train_scene)

    if args.white_bkgd:
            images = images[...,:3] * images[...,-1:] + (1. - images[...,-1:])
    else:
            images = images[...,:3]

    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    print('Load from iter: ', start)

    global_step = start

    bds_dict = {
        'near' : 2.,
        'far' : 6.,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    images = jt.array(images)
    poses = jt.array(poses)
    render_poses = jt.array(render_poses)

    N_rgb = args.N_rand
    if args.entropy:
        N_entropy = args.N_entropy
        loss_entropy = EntropyLoss(args)
    
    if args.render_only:
        print('RENDER ONLY')
        with jt.no_grad():
            images = None
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, savedir=testsavedir, render_factor=0)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'rgb.mp4'), to_byte(rgbs), fps=30, quality=8)
            disps[np.isnan(disps)] = 0
            print('Depth stats', np.mean(disps), np.max(disps), np.percentile(disps, 95))
            imageio.mimwrite(os.path.join(testsavedir, 'disp.mp4'), to_byte(disps / np.percentile(disps, 95)), fps=30, quality=8)
            return
    if args.render_test:
        testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', poses[test_index].shape)
        with jt.no_grad():
            rgbs, disps = render_path(jt.array(poses[test_index]), hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
        print('Saved test set')

        test_loss = calc_mse(jt.array(rgbs), images[test_index])
        test_psnr = calc_psnr(test_loss)
        print('test loss', test_loss.item(), 'test psnr', test_psnr.item())
        return
    
    print('Begin')
    print('TRAIN views are', train_index)
    print('TEST views are', test_index)
    print('VAL views are', val_index)
    for i in trange(start, args.N_iters+1):
        index = np.random.choice(train_index)
        target = images[index]
        pose = poses[index, :3, :4]
        rays_o, rays_d = get_rays(H, W, focal, jt.array(pose))  # (H, W, 3), (H, W, 3)
        if args.N_rand is not None:
            # uniform sampling
            if i < args.precrop_iters:
                dH = int(H//2 * args.precrop_frac)
                dW = int(W//2 * args.precrop_frac)
                coords = jt.stack(jt.meshgrid(jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)), -1)
            else:
                coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = jt.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = jt.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            if args.entropy and (args.N_entropy !=0):
                index = np.random.choice(len(images))
                target = images[index]
                pose = poses[index, :3,:4]
                rays_o, rays_d = get_rays(H, W, focal, jt.array(pose))  # (H, W, 3), (H, W, 3)
                
                if i < args.precrop_iters:
                        dH = int(H//2 * args.precrop_frac)
                        dW = int(W//2 * args.precrop_frac)
                        coords = jt.stack(jt.meshgrid(jt.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), jt.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)), -1)
                else:
                        coords = jt.stack(jt.meshgrid(jt.linspace(0, H-1, H), jt.linspace(0, W-1, W)), -1)  # (H, W, 2)
                
                coords = jt.reshape(coords, [-1,2])  # (H * W, 2)
                select_inds = np.random.choice(coords.shape[0], size=[N_entropy], replace=False)  # (N_entropy,)
                select_coords = coords[select_inds].long()  # (N_entropy, 2)
                rays_o_ent = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_entropy, 3)
                rays_d_ent = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_entropy, 3)
                batch_rays_entropy = jt.stack([rays_o_ent, rays_d_ent], 0) # (2, N_entropy, 3)
        N_rgb = batch_rays.shape[1]

        if args.entropy and (args.N_entropy !=0):
            batch_rays = jt.concat([batch_rays, batch_rays_entropy], 1)
        
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True, **render_kwargs_train)
        if args.entropy:
            acc_raw = acc 
            alpha_raw = extras['alpha']
        extras = {x:extras[x][:N_rgb] for x in extras}

        rgb = rgb[:N_rgb, :]
        disp = disp[:N_rgb] 
        acc = acc[:N_rgb]
        
        optimizer.zero_grad()
        img_loss = calc_mse(rgb, target_s)
        logging_info = {'rgb_loss': img_loss} 
        entropy_ray_zvals_loss = 0

        if args.entropy:
            entropy_ray_zvals_loss = loss_entropy.ray_zvals(alpha_raw, acc_raw)
            logging_info['entropy_ray_zvals'] = entropy_ray_zvals_loss
        
        loss = img_loss + args.entropy_ray_zvals_lambda * entropy_ray_zvals_loss 
        psnr = calc_psnr(img_loss)
        logging_info['psnr'] = psnr

        if 'rgb0' in extras:
            img_loss0 = calc_mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = calc_psnr(img_loss0)
            logging_info['rgb0_loss'] = img_loss0
            logging_info['psnr0'] = psnr0
        
        
        # loss.backward()
        optimizer.backward(loss)
        optimizer.step()

        # lr decay
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lr = args.lr * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

        # logging
        if (not jt.mpi or jt.mpi.local_rank()==0):
            # save model
            if i % 10000 == 0:
                jt.save({'global_step': i,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict() ,
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict() ,
                'optimizer_state_dict': optimizer.state_dict()}, 
                os.path.join(basedir, expname, '{:06d}.tar'.format(i)))
            
            # print log
            if i % 100 ==0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

            # save video
            if (args.i_video > 0 and i % args.i_video == 0 and i > 0):            
                with jt.no_grad():
                        rgbs, disps = render_path(render_poses, hwf, args.chunk, render_kwargs_test, render_factor=2)
                print('Done, saving', rgbs.shape, disps.shape)
                moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname.split('/')[-1], i))
                imageio.mimwrite(moviebase + 'rgb.mp4', to_byte(rgbs), fps=30, quality=8)
                imageio.mimwrite(moviebase + 'disp.mp4', to_byte(disps / np.nanmax(disps)), fps=30, quality=8)
            
            # test
            if (i%args.i_testset==0 ) and (i > 0) and (len(test_index) > 0):
                testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
                os.makedirs(testsavedir, exist_ok=True)
                print('test poses shape', poses[test_index].shape)
                with jt.no_grad():
                    rgbs, disps = render_path(jt.array(poses[test_index]), hwf, args.chunk, render_kwargs_test, savedir=testsavedir)
                print('Saved test set')

                test_loss = calc_mse(jt.array(rgbs), images[test_index])
                test_psnr = calc_psnr(test_loss)
                print('test loss', test_loss.item(), 'test psnr', test_psnr.item())

                                
                                
                        
if __name__ == '__main__':
    np.random.seed(0)
    jt.set_global_seed(0)
    jt.flags.use_cuda = 1
    args = parse_config()
    train(args)