# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
import torchmetrics

import distributed_util as dist_util
from evaluation import build_resnet50

from . import util
from .network import Image256Net
from .diffusion import Diffusion
from pathlib import Path
import torchvision.utils as vutils
from einops import rearrange
from skimage import img_as_ubyte, img_as_float32
import cv2
import scipy.io as scio

from ipdb import set_trace as debug


def build_optimizer_sched(opt, net, log):

    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
        log.info(f"[Opt] Built lr step scheduler {sched_dict=}!")
    else:
        sched = None

    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log)
    return torch.cat(gathered_t).detach().cpu()

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def imwrite(im_in, path, chn='rgb', dtype_in='float32', qf=None):
    '''
    Save image.
    Input:
        im: h x w x c, numpy tensor
        path: the saving path
        chn: the channel order of the im,
    '''
    im = im_in.copy()
    if isinstance(path, str):
        path = Path(path)
    if dtype_in != 'uint8':
        im = img_as_ubyte(im)

    if chn.lower() == 'rgb' and im.ndim == 3:
        im = rgb2bgr(im)

    if qf is not None and path.suffix.lower() in ['.jpg', '.jpeg']:
        flag = cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    else:
        flag = cv2.imwrite(str(path), im)

    return flag

class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # Save opt.
        if save_opt and opt.phase!='test':
            opt_pkl_path = os.path.join(opt.ckpt_path, "options.pkl")
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        betas = np.concatenate([betas[:opt.interval//2], np.flip(betas[:opt.interval//2])])
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Built I2SB diffusion: steps={len(betas)}!")

        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        self.net = Image256Net(log, noise_levels=noise_levels, use_fp16=opt.use_fp16, cond=opt.cond_x1, ckpt_dir=os.path.join(opt.root_path, "data"))
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)

        self.log = log

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def sample_batch(self, opt, batch):
        if opt.corrupt == "mixture": 
            if opt.phase=='test' or opt.phase=='val':
                if opt.cond_x1:
                    clean_img, corrupt_img, cond_img = batch[0], batch[1], batch[2]
                else:
                    clean_img, corrupt_img = batch[0], batch[1]
                mask = None
            else:
                if opt.cond_x1:
                    clean_img, corrupt_img, cond_img = next(batch)
                else:
                    clean_img, corrupt_img = next(batch)
                mask = None
        # os.makedirs(".debug", exist_ok=True)
        # tu.save_image((clean_img+1)/2, ".debug/clean.png", nrow=4)
        # tu.save_image((corrupt_img+1)/2, ".debug/corrupt.png", nrow=4)
        # debug()

        x0 = clean_img.detach().to(opt.device)
        x1 = corrupt_img.detach().to(opt.device)
        if opt.cond_x1:
            c0 = cond_img.detach().to(opt.device)  
        if mask is not None:
            mask = mask.detach().to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
        cond = c0.detach() if opt.cond_x1 else None

        if opt.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, cond

    def train(self, opt, train_dataset, val_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema
        optimizer, sched = build_optimizer_sched(opt, net, log)

        train_loader = util.setup_loader(train_dataset, opt.microbatch, shuffle=True)
        val_loader   = util.DataLoaderX(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=4,
                                    drop_last=True)

        self.accuracy = torchmetrics.Accuracy().to(opt.device)
        self.resnet = build_resnet50().to(opt.device)

        net.train()
        n_inner_loop = opt.batch_size // (opt.global_size * opt.microbatch)
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            for _ in range(n_inner_loop):
                # ===== sample boundary pair =====
                
                opt.phase='train'
                x0, x1, mask, cond = self.sample_batch(opt, train_loader)
    
                # ===== compute loss =====
                step = torch.randint(0, opt.interval, (x0.shape[0],))

                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
                label = self.compute_label(step, x0, xt)

                pred = net(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape

                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)
                loss.backward()

            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            if it % 500 == 0:
                if opt.global_rank == 0:
                    torch.save({
                        "net": self.net.state_dict(),
                        "ema": ema.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "sched": sched.state_dict() if sched is not None else sched,
                    }, os.path.join(opt.ckpt_path, "latest{}.pt".format(it)))
                    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")
                if opt.distributed:
                    torch.distributed.barrier()

            if it == 500 or it % 500 == 0: # 0, 0.5k, 3k, 6k 9k
                net.eval()
                opt.phase='val'
                self.evaluation(opt, it, val_loader)
                net.train()
        self.writer.close()
            
    
    def test(self, opt, test_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema

        test_loader = util.DataLoaderX(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=4,
                                    drop_last=True)

        net.eval()
        it = 0
        self.evaluation(opt, it, test_loader)

    def test_sampling(self, opt, test_dataset):
        self.writer = util.build_log_writer(opt)
        log = self.log

        net = DDP(self.net, device_ids=[opt.device])
        ema = self.ema

        test_loader = util.DataLoaderX(test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=4,
                                    drop_last=True)

        net.eval()
        it = 0
        self.evaluation_sampling(opt, it, test_loader)
        
    
    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or opt.interval-1
        assert 0 < nfe < opt.interval == len(self.diffusion.betas)
        steps = util.space_indices(opt.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(opt.device)
        if cond is not None: cond = cond.to(opt.device)
        if mask is not None:
            mask = mask.to(opt.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                out = self.net(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0
    
    @torch.no_grad()
    def evaluation(self, opt, it, val_loader):

        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        for iteration, batch in enumerate(val_loader, 1):
            img_clean, img_corrupt, mask, cond = self.sample_batch(opt, batch)

            x1 = img_corrupt.to(opt.device)

            xs, pred_x0s = self.ddpm_sampling(
                opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
            )

            log.info("Collecting tensors ...")
            img_clean   = all_cat_cpu(opt, log, img_clean)
            img_corrupt = all_cat_cpu(opt, log, img_corrupt)
            xs          = all_cat_cpu(opt, log, xs)
            pred_x0s    = all_cat_cpu(opt, log, pred_x0s)

            batch, len_t, *xdim = xs.shape
            assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
            assert xs.shape == pred_x0s.shape
            log.info(f"Generated recon trajectories: size={xs.shape}")

            def log_image(tag, img, nrow=10):
                self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

            """
            def log_accuracy(tag, img):
                pred = self.resnet(img.to(opt.device)) # input range [-1,1]
                accu = self.accuracy(pred, y.to(opt.device))
                self.writer.add_scalar(it, tag, accu)
            """

            log.info("Logging images ...")
            img_recon = xs[:, 0, ...]
            log_image("image/clean",   img_clean)
            log_image("image/corrupt", img_corrupt)
            log_image("image/recon",   img_recon)
            log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
            log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)
            
            im_tensor = torch.cat((torch.flip(xs[:, range(0,len_t,2), ...].squeeze(2),dims=[1]), img_clean), dim=1)
            #im_tensor = im_tensor.permute(0,1,3,2)
            im_tensor = rearrange(im_tensor, 'b (k c) h w -> (b k) c h w', c=img_corrupt.shape[1])
                        
            im_tensor = vutils.make_grid(im_tensor, nrow=6, normalize=True, scale_each=True) # c x H x W
            im_path = os.path.join(opt.image_path, f"val-{it}-{iteration-1}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            imwrite(im_np, im_path)

            """
            log.info("Logging accuracies ...")
            log_accuracy("accuracy/clean",   img_clean)
            log_accuracy("accuracy/corrupt", img_corrupt)
            log_accuracy("accuracy/recon",   img_recon)
            """
            if opt.phase=='test':
                E_img = img_recon.squeeze(0).squeeze(0).detach().float().cpu().numpy()
                save_mat_path = os.path.join(opt.mat_dir, '2D_brain_{}.mat'.format( iteration-1))
                scio.savemat(save_mat_path, {'superRes': E_img})
                print('Testing I2SB: {}'.format(iteration-1))


            log.info(f"========== Evaluation finished: iter={iteration} ==========")
            torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluation_sampling(self, opt, it, val_loader):

        log = self.log
        
        for iteration, batch in enumerate(val_loader, 1):
            img_clean, img_corrupt, mask, cond = self.sample_batch(opt, batch)
            
            for num_sample in range(0, opt.num_sampling):    
                log.info(f"========== Evaluation started: iter={iteration-1+opt.start_test_index}, sample={num_sample} ==========")
                x1 = img_corrupt.to(opt.device)

                xs, pred_x0s = self.ddpm_sampling(
                    opt, x1, mask=mask, cond=cond, clip_denoise=opt.clip_denoise, verbose=opt.global_rank==0
                )

                log.info("Collecting tensors ...")
                img_clean   = all_cat_cpu(opt, log, img_clean)
                img_corrupt = all_cat_cpu(opt, log, img_corrupt)
                xs          = all_cat_cpu(opt, log, xs)
                pred_x0s    = all_cat_cpu(opt, log, pred_x0s)
        
                batch, len_t, *xdim = xs.shape
                assert img_clean.shape == img_corrupt.shape == (batch, *xdim)
                assert xs.shape == pred_x0s.shape
                log.info(f"Generated recon trajectories: size={xs.shape}")

                def log_image(tag, img, nrow=10):
                    self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow)) # [1,1] -> [0,1]

                """
                def log_accuracy(tag, img):
                    pred = self.resnet(img.to(opt.device)) # input range [-1,1]
                    accu = self.accuracy(pred, y.to(opt.device))
                    self.writer.add_scalar(it, tag, accu)
                """

                log.info("Logging images ...")
                img_recon = xs[:, 0, ...]
                log_image("image/clean",   img_clean)
                log_image("image/corrupt", img_corrupt)
                log_image("image/recon",   img_recon)
                log_image("debug/pred_clean_traj", pred_x0s.reshape(-1, *xdim), nrow=len_t)
                log_image("debug/recon_traj",      xs.reshape(-1, *xdim),      nrow=len_t)
                if num_sample<50:
                    im_tensor = torch.cat((torch.flip(xs[:, range(0,len_t,2), ...].squeeze(2),dims=[1]), img_clean), dim=1)
                    #im_tensor = im_tensor.permute(0,1,3,2)
                    im_tensor = rearrange(im_tensor, 'b (k c) h w -> (b k) c h w', c=img_corrupt.shape[1])
                                
                    im_tensor = vutils.make_grid(im_tensor, nrow=6, normalize=True, scale_each=True) # c x H x W
                    im_path = os.path.join(opt.image_path, f"val-{iteration-1+opt.start_test_index}-{num_sample}.png")
                    im_np = im_tensor.cpu().permute(1,2,0).numpy()
                    imwrite(im_np, im_path)

                """
                log.info("Logging accuracies ...")
                log_accuracy("accuracy/clean",   img_clean)
                log_accuracy("accuracy/corrupt", img_corrupt)
                log_accuracy("accuracy/recon",   img_recon)
                """
                if opt.phase=='test':
                    E_img = img_recon.squeeze(0).squeeze(0).detach().float().cpu().numpy()
                    L_img = img_corrupt.squeeze(0).squeeze(0).detach().float().cpu().numpy()
                    #E_img = np.transpose(E_img, (1,0))
                    
                    #save_mat_path = os.path.join(opt.mat_dir, '2D_brain_{}_{}.mat'.format(iteration-1, num_sample))
                    save_mat_path = os.path.join(opt.mat_dir, '2D_brain_{}_{}.mat'.format(iteration-1+opt.start_test_index, num_sample))
                    scio.savemat(save_mat_path, {'superRes': E_img})
                    print('Testing I2SB on {}: {}-{}'.format(opt.dataset, iteration-1+opt.start_test_index, num_sample))


                log.info(f"========== Evaluation finished: iter={it} ==========")
                torch.cuda.empty_cache()

