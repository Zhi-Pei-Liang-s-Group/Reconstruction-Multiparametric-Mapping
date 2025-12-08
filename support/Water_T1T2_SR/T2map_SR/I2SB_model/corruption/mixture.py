# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import numpy as np
import enum
import os
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import normalize

from .jpeg import jpeg_decode, jpeg_encode
from .blur import Deblurring
from .superresolution import build_sr_bicubic, build_sr_pool
from .inpaint import get_center_mask, load_freeform_masks
import mat73
import pandas as pd

from ipdb import set_trace as debug
import scipy.io as scio


class AllCorrupt(enum.IntEnum):
    JPEG_5 = 0
    JPEG_10 = 1
    BLUR_UNI = 2
    BLUR_GAUSS = 3
    SR4X_POOL = 4
    SR4X_BICUBIC = 5
    INPAINT_CENTER = 6
    INPAINT_FREE1020 = 7
    INPAINT_FREE2030 = 8

class MixtureCorruptMethod:
    def __init__(self, opt, device="cpu"):

        # ===== blur ====
        self.blur_uni = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, opt.image_size, device)

        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, device)

        # ===== sr4x ====
        factor = 4
        self.sr4x_pool = build_sr_pool(factor, device, opt.image_size)
        self.sr4x_bicubic = build_sr_bicubic(factor, device, opt.image_size)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')

        # ===== inpaint ====
        self.center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1, 1, 256, 256]
        self.free1020_masks = torch.from_numpy((load_freeform_masks("freeform1020"))) # [10000, 1, 256, 256]
        self.free2030_masks = torch.from_numpy((load_freeform_masks("freeform2030"))) # [10000, 1, 256, 256]

    def jpeg(self, img, qf):
        return jpeg_decode(jpeg_encode(img, qf), qf)

    def blur(self, img, kernel):
        img = (img + 1) / 2
        if kernel == "uni":
            _img = self.blur_uni.H(img).reshape(*img.shape)
        elif kernel == "gauss":
            _img = self.blur_gauss.H(img).reshape(*img.shape)
        # [0,1] -> [-1,1]
        return _img * 2 - 1

    def sr4x(self, img, filter):
        b, c, w, h = img.shape
        if filter == "pool":
            _img = self.sr4x_pool.H(img).reshape(b, c, w // 4, h // 4)
        elif filter == "bicubic":
            _img = self.sr4x_bicubic.H(img).reshape(b, c, w // 4, h // 4)

        # scale to original image size for I2SB
        return self.upsample(_img)

    def inpaint(self, img, mask_type, mask_index=None):
        if mask_type == "center":
            mask = self.center_mask
        elif mask_type == "free1020":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free1020_masks))
            mask = self.free1020_masks[[mask_index]]
        elif mask_type == "free2030":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free2030_masks))
            mask = self.free2030_masks[[mask_index]]
        return img * (1. - mask) + mask * torch.randn_like(img)

    def mixture(self, img, corrupt_idx, mask_index=None):
        if corrupt_idx == AllCorrupt.JPEG_5:
            corrupt_img = self.jpeg(img, 5)
        elif corrupt_idx == AllCorrupt.JPEG_10:
            corrupt_img = self.jpeg(img, 10)
        elif corrupt_idx == AllCorrupt.BLUR_UNI:
            corrupt_img = self.blur(img, "uni")
        elif corrupt_idx == AllCorrupt.BLUR_GAUSS:
            corrupt_img = self.blur(img, "gauss")
        elif corrupt_idx == AllCorrupt.SR4X_POOL:
            corrupt_img = self.sr4x(img, "pool")
        elif corrupt_idx == AllCorrupt.SR4X_BICUBIC:
            corrupt_img = self.sr4x(img, "bicubic")
        elif corrupt_idx == AllCorrupt.INPAINT_CENTER:
            corrupt_img = self.inpaint(img, "center")
        elif corrupt_idx == AllCorrupt.INPAINT_FREE1020:
            corrupt_img = self.inpaint(img, "free1020", mask_index=mask_index)
        elif corrupt_idx == AllCorrupt.INPAINT_FREE2030:
            corrupt_img = self.inpaint(img, "free2030", mask_index=mask_index)
        return corrupt_img


class MixtureCorruptDatasetTrain(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetTrain, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        rand_idx = np.random.choice(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), rand_idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img
    
    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

class CorruptDataset_2DBrain(Dataset):
    def __init__(self, opt, phase):
        super(CorruptDataset_2DBrain, self).__init__()
        if phase=='train' :
            self.start_index  = 0
            self.dataset  = opt.dataset_dir_train
            self.data_len = len(os.listdir(self.dataset))
        elif phase=='val':
            self.start_index  = opt.start_test_index
            self.dataset  = opt.dataset_dir_val
            self.data_len = len(os.listdir(self.dataset))
        elif phase=='test':
            self.start_index  = opt.start_test_index
            self.dataset  = opt.dataset_dir_test
            self.data_len = len(os.listdir(self.dataset))#+opt.start_test_index   
        self.phase        = phase
        self.mean         = 0.5
        self.std          = 0.5
        
        self.have_GT      = opt.have_GT
        #self.round_str    = opt.round_str

    def load_mat_file(self, filepath, var):
        try:
            # Try using scipy first (works with MATLAB v7.2 and below)
            data = scio.loadmat(filepath)[var]
            #print("Loaded using scipy.io.loadmat")
        except NotImplementedError:
            try:
                # If it's MATLAB v7.3 (HDF5 format), use mat73
                data = mat73.loadmat(filepath)[var]
                #print("Loaded using mat73.loadmat")
            except Exception as e:
                print(f"Failed to load .mat file: {e}")
                data = None
        return data
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        mat_file_path = os.path.join(self.dataset,'2D_brain_'+str(index+self.start_index)+'.mat')

        lowRes = self.load_mat_file(mat_file_path, 'lowRes')
        if self.phase=='train' or self.phase=='val' or self.have_GT:
            highRes = self.load_mat_file(mat_file_path, 'highRes')
        else:
            highRes = lowRes
   
        highRes = highRes[:,:,np.newaxis]
        lowRes  = lowRes[:,:,np.newaxis]

        highRes_min = highRes.min()  
        highRes_max = highRes.max()  
        highRes = (highRes - highRes_min) / (highRes_max - highRes_min) 

        lowRes_min = lowRes.min()  
        lowRes_max = lowRes.max()  
        lowRes = (lowRes - lowRes_min) / (lowRes_max - lowRes_min) 

        img_gt, img_lq = img2tensor([highRes, lowRes], bgr2rgb=False, float32=True)

        if self.mean is not None or self.std is not None:
            img_lq = normalize(img_lq, self.mean, self.std, inplace=True)
            img_gt = normalize(img_gt, self.mean, self.std, inplace=True)
           
        assert img_lq.shape == img_gt.shape, (img_gt.shape, img_lq.shape)

        return img_gt, img_lq

class CorruptDataset_2DBrain_csv(Dataset):
    def __init__(self, opt, phase):
        super(CorruptDataset_2DBrain_csv, self).__init__()
        if phase=='train':
            self.dataset  = opt.train_csv
            self.df       = pd.read_csv(self.dataset, header = None)
            self.data_len = len(self.df)-1
        elif phase=='val':
            self.dataset  = opt.dataset_dir_val
            self.data_len = len(os.listdir(self.dataset))
        elif phase=='test':
            self.dataset  = opt.dataset_dir_test
            self.data_len = len(os.listdir(self.dataset))-opt.start_test_index   
        self.phase        = phase
        self.mean         = 0.5
        self.std          = 0.5
        self.start_index  = opt.start_test_index

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        mat_file_path = list(self.df[3])[index+1]
        #mat_file_path = os.path.join(self.dataset,'2D_brain_'+str(index+self.start_index)+'.mat')
        lowRes        = mat73.loadmat(mat_file_path)['lowRes'].astype(np.float32)
        
        if self.phase=='train' or self.phase=='val' or 'v2' in str(self.dataset):
            highRes = mat73.loadmat(mat_file_path)['superRes'].astype(np.float32)
        else:
            highRes = lowRes
   
        highRes = highRes[:,:,np.newaxis]
        lowRes  = lowRes[:,:,np.newaxis]

        highRes_min = highRes.min()  
        highRes_max = highRes.max()  
        highRes = (highRes - highRes_min) / (highRes_max - highRes_min) 

        lowRes_min = lowRes.min()  
        lowRes_max = lowRes.max()  
        lowRes = (lowRes - lowRes_min) / (lowRes_max - lowRes_min) 

        img_gt, img_lq = img2tensor([highRes, lowRes], bgr2rgb=False, float32=True)

        if self.mean is not None or self.std is not None:
            img_lq = normalize(img_lq, self.mean, self.std, inplace=True)
            img_gt = normalize(img_gt, self.mean, self.std, inplace=True)

        assert img_lq.shape == img_gt.shape, (img_gt.shape, img_lq.shape)

        return img_gt, img_lq

class MixtureCorruptDatasetVal(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetVal, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        idx = index % len(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), idx, mask_index=idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y
