#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run I2SB super-resolution inference for T2 maps on the MRx demo data.

Usage example:
    python run_T2map_I2SB_SR_demo.py \
        --subject-id MRx_uploaded_demo_data \
        --dataset-name SPICE_352x352_from_214x122
"""

import os
import copy
import sys
import argparse
from pathlib import Path

import torch
from torch.multiprocessing import Process

#THIS_DIR = Path('/share/projectdata/SPICE_Water_Super_Resolution/code/Water_T1T2_SR').resolve()
#I2SB_DIR = THIS_DIR / "T1map_SR" / "I2SB_model" 
#sys.path.insert(0, str(I2SB_DIR))
                
from distributed_util import init_processes
from test import create_training_options, main

def parse_args():
    parser = argparse.ArgumentParser(description="Run T2map I2SB SR inference.")

    parser.add_argument(
        "--subject-id",
        type=str,
        #required=True,
        default='MRx_uploaded_demo_data',
        help="Subject ID (folder name under Data/Testing_data).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        #required=True,
        default='SPICE_352x352_from_214x122',
        help="Dataset name (folder name under subject/LR_data).",
    )
    return parser.parse_args()


def run_demo(subject_id: str, dataset_name: str):
    # ------------------------------------------------------------------
    # Paths: root_path = .../Water_T1T2_SR/T1map_SR/I2SB_model
    # ------------------------------------------------------------------
    model_path = Path(__file__).resolve().parent

    # Data root: .../Water_T1T2_SR/T1map_SR/Data/Testing_data
    data_root = model_path.parent / "Testing_data"

    # ------------------------------------------------------------------
    # Experiment configuration (change these if using a different model)
    # ------------------------------------------------------------------
    # This must match the experiment folder that contains the checkpoint.
    # For example, if your trained experiment directory is:
    #   <root_path>/experiment_base_T1map_SR_demo/ckpt/latest4500.pt
    # then set:
    #   model_name = "experiment_base_T1map_SR_demo"
    model_name = "experiment_base"

    start_index = 0          # index of the first 2D_brain_x.mat to test
    gpu_id      = 0          # GPU to use
    master_port = "6021"

    # ------------------------------------------------------------------
    # Prepare options for this subject
    # ------------------------------------------------------------------
    data_path = data_root / subject_id

    opt = create_training_options(
        root_path=str(model_path),
        data_path=str(data_path),
        dataset_name=dataset_name,
        model_name=model_name,
        start_index=start_index,
        gpu=gpu_id,
        port=master_port,
    )

    assert opt.corrupt is not None

    # ------------------------------------------------------------------
    # Single-GPU (non-distributed) inference by default
    # ------------------------------------------------------------------
    if opt.distributed:
        size = opt.n_gpu_per_node
        processes = []

        for rank in range(size):
            opt_i = copy.deepcopy(opt)
            opt_i.local_rank = rank
            global_rank = rank + opt_i.node_rank * opt_i.n_gpu_per_node
            global_size = opt_i.num_proc_node * opt_i.n_gpu_per_node
            opt_i.global_rank = global_rank
            opt_i.global_size = global_size
            print(
                "Node rank %d, local proc %d, global proc %d, global_size %d"
                % (opt_i.node_rank, rank, global_rank, global_size)
            )
            p = Process(
                target=init_processes,
                args=(global_rank, global_size, main, opt_i),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # Non-distributed: use a single GPU
        torch.cuda.set_device(opt.gpu)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        init_processes(0, opt.n_gpu_per_node, main, opt)


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.subject_id, args.dataset_name)
