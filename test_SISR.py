import os
import sys
import time
import math
import cv2

from torch.autograd import Variable
from torchvision.transforms import ToPILImage, ToTensor

from utils import *

import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='spsr')
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_G_path', type=str, default='/group/20012/jinnieli/SPSR_pth/experiments/SPSR/models/72600_G.pth')
    parser.add_argument('--self_ensemble', type=bool, default=True)
    parser.add_argument('--mode', type=str, default='CNA')
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=23)
    parser.add_argument('--in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--gc', type=int, default=32)
    parser.add_argument('--group', type=int, default=1)
    return parser.parse_args()

def test_SISR(cfg, model, LR_img):
    #cfg = parse_args()
    # Create model
    #model = SPSRModel(cfg)
    #model.to(cfg.device)

    if cfg.self_ensemble:
        sr_images = []
        for flip_index in range(2):  # for flipping

            LR_img = np.transpose(LR_img, axes=(1, 0, 2))

            for rotate_index in range(4):  # for rotating
                
                LR_img = np.rot90(LR_img, k=1, axes=(0, 1))

                # if rotate_index == 0 or rotate_index == 2:
                #     continue

                LR_img = np.ascontiguousarray(LR_img)

                LR_img = torch.from_numpy(LR_img.transpose((2, 0, 1)))
                LR_img = LR_img.unsqueeze(0).to(cfg.device)
                
                #print(LR_img.max())
                # test
                #print(LR_img.shape)
                SR_img = model.test(LR_img)
                SR_img = SR_img.detach()[0].float().cpu()
                #print(SR_img.max())
                
                SR_img = torch.clamp(SR_img, 0, 1)
                SR_img = np.transpose(torch.squeeze(SR_img, 0).numpy(), axes=(1, 2, 0))  #torch.squeeze(SR_img.data.cpu(), 0).numpy()

                LR_img = np.transpose(torch.squeeze(LR_img.data.cpu(), 0).numpy(), axes=(1, 2, 0))

                SR_img = np.rot90(SR_img, k=(3 - rotate_index), axes=(0, 1))

                if (flip_index == 0):
                    SR_img = np.transpose(SR_img, axes=(1, 0, 2))
                
                #save_img(SR_img, './Figs/'  + str(flip_index) + '_' + str(rotate_index) + '_SISR.png')
                #tmp_save_sr = ToPILImage()((SR_img * 255).astype('uint8'))
                #tmp_save_sr.save(
                #     )
                # tmp_save_right = transforms.ToPILImage()((SR_right * 255).astype('uint8'))
                # tmp_save_right.save(
                #     save_path + '/' + str(idx).zfill(4) + '_' + str(flip_index) + '_' + str(rotate_index) + '_R.png')
                
                #SR_img = np.clip(SR_img/255, 0, 1)
                #print(SR_img.shape)
                #print('************')
                sr_images.append(SR_img)
    else:
        print('******************')

    sr_images = np.mean(sr_images, axis=0)
    #sr_images = np.clip(sr_images, 0, 1)
    #print(sr_images.shape)
    return sr_images