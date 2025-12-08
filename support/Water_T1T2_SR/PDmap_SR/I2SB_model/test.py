# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.multiprocessing import Process

from logger import Logger
from distributed_util import init_processes
from corruption import build_corruption
from dataset import imagenet
from i2sb import Runner, download_ckpt

import colored_traceback.always
from ipdb import set_trace as debug
import corruption.mixture as mix

RESULT_DIR = Path("results")

def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

def create_training_options(root_path, data_path, dataset_name, model_name, start_index, gpu=3, port='6021'):
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",                 type=int,   default=0)
    parser.add_argument("--root_path",            type=str,   default=root_path,        help="address for the project")
    parser.add_argument("--name",                 type=str,   default=model_name,        help="experiment ID")
    parser.add_argument("--ckpt",                 type=bool,  default=True,        help="resumed checkpoint name")
    parser.add_argument("--gpu",                  type=int,   default=gpu,         help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node",       type=int,   default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address",       type=str,   default='localhost', help="address for master")
    parser.add_argument("--master-port",          type=str,   default=port,        help="port for master")
    parser.add_argument("--node-rank",            type=int,   default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",        type=int,   default=1,           help="The number of nodes in multi node env")
    # parser.add_argument("--amp",                action="store_true")

    # --------------- SB model ---------------
    parser.add_argument("--image-size",           type=int,   default=256)
    parser.add_argument("--corrupt",              type=str,   default='mixture',        help="restoration task")
    parser.add_argument("--t0",                   type=float, default=1e-4,             help="sigma start time in network parametrization")
    parser.add_argument("--T",                    type=float, default=1.,               help="sigma end time in network parametrization")
    parser.add_argument("--interval",             type=int,   default=100,              help="number of interval")
    parser.add_argument("--beta-max",             type=float, default=0.3,              help="max diffusion for the diffusion model")
    # parser.add_argument("--beta-min",           type=float, default=0.1)
    parser.add_argument("--ot-ode",               action="store_true",                  help="use OT-ODE model")
    parser.add_argument("--clip-denoise",         action="store_true", default=True,    help="clamp predicted image to [-1,1] at each")

    # optional configs for conditional network
    parser.add_argument("--cond-x1",              action="store_true", default=False,    help="conditional the network on degraded images")
    parser.add_argument("--add-x1-noise",         action="store_true",                  help="add noise to conditional network")

    # --------------- optimizer and loss ---------------
    parser.add_argument("--batch-size",           type=int,   default=1)
    parser.add_argument("--microbatch",           type=int,   default=1,           help="accumulate gradient over microbatch until full batch-size")
    parser.add_argument("--num-itr",              type=int,   default=1,           help="training iteration")
    parser.add_argument("--lr",                   type=float, default=5e-5,        help="learning rate")
    parser.add_argument("--lr-gamma",             type=float, default=0.99,        help="learning rate decay ratio")
    parser.add_argument("--lr-step",              type=int,   default=1000,        help="learning rate decay step size")
    parser.add_argument("--l2-norm",              type=float, default=0.0)
    parser.add_argument("--ema",                  type=float, default=0.99)

    # --------------- path and logging ---------------
    parser.add_argument("--dataset",              type=str,   default=dataset_name,     help="2D Brain test dataset")
    parser.add_argument("--dataset-dir-test",     type=Path,  default=data_path,  help="path to 2D Brain test dataset")
    parser.add_argument("--result-dir-test",      type=Path,  default=data_path,  help="path to 2D Brain test dataset")
    parser.add_argument("--start-test-index",     type=int,   default=start_index,          help="start index for test data")
    parser.add_argument("--log-dir",              type=Path,  default="log",         help="path to log std outputs and writer data")
    parser.add_argument("--ckpt-dir",             type=Path,  default="ckpt",        help="path to save the trained model")
    parser.add_argument("--image-dir",            type=Path,  default="image",       help="path to save the validation image")
    parser.add_argument("--log-writer",           type=str,   default=None,          help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",        type=str,   default=None,          help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",           type=str,   default=None,          help="user name of your W&B account")
    parser.add_argument(
        "--subject-id",
        type=str,
        required=True,
        #default='MRx_uploaded_demo_data',
        help="Subject ID (folder name under Data/Testing_data).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        #default='SPICE_352x352_from_214x122',
        help="Dataset name (folder name under subject/LR_data).",
    )


    opt = parser.parse_args()

    opt.phase = 'test'
    opt.num_sampling = 1


    # ========= auto setup =========
    opt.device='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    if opt.name is None:
        opt.name = opt.corrupt
    opt.distributed = opt.n_gpu_per_node > 1
    opt.use_fp16 = False # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    if opt.cond_x1:
        opt.name       = opt.name+'_'+str(opt.interval)+'_cond_x1'
    else:
        opt.name       = opt.name+'_'+str(opt.interval)
    opt.log_dir    = os.path.join(opt.root_path, opt.name, opt.log_dir)
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path  = os.path.join(opt.root_path, opt.name, opt.ckpt_dir)
    os.makedirs(opt.ckpt_path, exist_ok=True)

    opt.image_path = os.path.join(str(opt.result_dir_test), dataset_name, 'SR_data', 'image')
    os.makedirs(opt.image_path, exist_ok=True)
    opt.mat_dir    = os.path.join(str(opt.result_dir_test), dataset_name, 'SR_data', 'mat')
    os.makedirs(opt.mat_dir, exist_ok=True) 
    opt.dataset_dir_test =  os.path.join(str(opt.dataset_dir_test), dataset_name, 'LR_data', 'mat')

    opt.have_GT = True

    if opt.ckpt:
        ckpt_file = Path(os.path.join(opt.root_path,opt.name,opt.ckpt_path,"latest.pt"))
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None

    # ========= auto assert =========
    assert opt.batch_size % opt.microbatch == 0, f"{opt.batch_size=} is not dividable by {opt.microbatch}!"
    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("         Image-to-Image Schrodinger Bridge")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build 2D brain dataset
    test_dataset  = mix.CorruptDataset_2DBrain(opt, phase='test')

    # note: images should be normalized to [-1,1] for corruption methods to work properly

    # build corruption method
    # corrupt_method = build_corruption(opt, log)

    run = Runner(opt, log)
    run.test_sampling(opt, test_dataset)
    log.info("Finish!")

if __name__ == '__main__':

    subjectID_list = {'2025_11_02_invivo_R1'} #2025_11_05_invivo_R1, PrismaI_2025_08_04_R2.7_new, PrismaI_2025_07_29_R1.1, MRx_uploaded_demo_data
    
    for subjectID in subjectID_list: 
        root_path   = '/share/projectdata/SPICE_Water_Super_Resolution/code/Fundation_T1_SR/I2SB_transfer_learning_352x352'
        data_path   = '/share/projectdata/SPICE_Water_Super_Resolution/code/Fundation_T1_SR/Data/Testing_data/'+subjectID
        Dataset     = {'SPICE_352x352_from_214x122_denoise'}
        model_name  = 'experiment_base_Transfer_learning_352x352_from_214x122_denoise'
        start_index = 0

        method_name = 'I2SB'
        epoch = 4500 #6000 for y-x space SR;

        for dataset_name in Dataset:
            
            opt = create_training_options(root_path, data_path, dataset_name, model_name, start_index, gpu=7, port='6027')

            assert opt.corrupt is not None

            # one-time download: ADM checkpoint
            download_ckpt(os.path.join(opt.root_path, "data"))

            if opt.distributed:
                size = opt.n_gpu_per_node

                processes = []
                for rank in range(size):
                    opt = copy.deepcopy(opt)
                    opt.local_rank = rank
                    global_rank = rank + opt.node_rank * opt.n_gpu_per_node
                    global_size = opt.num_proc_node * opt.n_gpu_per_node
                    opt.global_rank = global_rank
                    opt.global_size = global_size
                    print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))
                    p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
                    p.start()
                    processes.append(p)

                for p in processes:
                    p.join()
            else:
                torch.cuda.set_device(0)
                opt.global_rank = 0
                opt.local_rank = 0
                opt.global_size = 1
                init_processes(0, opt.n_gpu_per_node, main, opt)
