import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torchvision.datasets as datasets

from model import VMMpp
from data_loader import ImageFromFolderTest
from utils import AverageMeter
import numpy as np
from PIL import Image
from collections import OrderedDict


def main(args):
    print(args)

    # create model
    model = VMMpp(img_size=384, patch_size=1, in_chans=3,
                 embed_dim=192, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 manipulator_num_resblk = 1).to(device)

    # load checkpoint
    if os.path.isfile(args.load_ckpt):
        print("=> loading checkpoint '{}'".format(args.load_ckpt))
        checkpoint = torch.load(args.load_ckpt)
        args.start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.load_ckpt, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.load_ckpt))
        assert(False)
        

    # check saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # data loader
    dataset_mag = ImageFromFolderTest(args.video_path, mag=args.amp, mode=args.mode, num_data=args.num_data, preprocessing=False) 
    data_loader = data.DataLoader(dataset_mag, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=False)
    

    # generate frames 
    mag_frames=[]
    model.eval()

    # static mode or dynamic mode
    if args.mode=='static' or args.mode=='dynamic':
        for i, (xa, xb, amp_factor) in enumerate(data_loader):
            if i%10==0: print('processing sample %d'%i)
            amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
 
            xa=xa.to(device)
            xb=xb.to(device)
            amp_factor=amp_factor.to(device)
 
            y_hat, _, _, _ = model(xa, xb, amp_factor)
           
            if i==0: 
                 # back to image scale (0-255) 
                 tmp = xa.permute(0,2,3,1).cpu().detach().numpy()
                 tmp = np.clip(tmp, -1.0, 1.0)
                 tmp = ((tmp + 1.0) * 127.5).astype(np.uint8)
                 mag_frames.append(tmp)
            
            # back to image scale (0-255) 
            y_hat = y_hat.permute(0,2,3,1).cpu().detach().numpy()
            y_hat = np.clip(y_hat, -1.0, 1.0)
            y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)
            mag_frames.append(y_hat)

    else:
        # temporal mode (difference of IIR)
        # copy filter coefficients and follow codes from https://github.com/12dmodel/deep_motion_mag 
        filter_b = [args.fh-args.fl, args.fl-args.fh]
        filter_a = [-1.0*(2.0 - args.fh - args.fl), (1.0 - args.fl) * (1.0 - args.fh)]

        x_state = []
        y_state = []
        for i, (xa, xb, amp_factor) in enumerate(data_loader):
            if i%10==0: print('processing sample %d'%i)
            amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
 
            xa=xa.to(device)
            xb=xb.to(device)
            amp_factor=amp_factor.to(device)
 
            vb, mb = model.encoder(xb)
            x_state.insert(0,mb.detach())
            while len(x_state)<len(filter_b):
                x_state.insert(0,mb.detach())
            if len(x_state)>len(filter_b):
                x_state = x_state[:len(filter_b)]
            y = torch.zeros_like(mb)
            for i in range(len(x_state)):
                y += x_state[i] * filter_b[i]
            for i in range(len(y_state)):
                y -= y_state[i] * filter_a[i]

            y_state.insert(0,y.detach())
            if len(y_state) > len(filter_a):
                y_state = y_state[:len(filter_a)]

            mb_m = model.manipulator(0.0, y, amp_factor)
            mb_m += mb - y

            y_hat = model.decoder(vb, mb_m)
             
            if i==0: 
                 # back to image scale (0-255) 
                 tmp = xa.permute(0,2,3,1).cpu().detach().numpy()
                 tmp = np.clip(tmp, -1.0, 1.0)
                 tmp = ((tmp + 1.0) * 127.5).astype(np.uint8)
                 mag_frames.append(tmp)
            
            # back to image scale (0-255) 
            y_hat = y_hat.permute(0,2,3,1).cpu().detach().numpy()
            y_hat = np.clip(y_hat, -1.0, 1.0)
            y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)
            mag_frames.append(y_hat)
            


    # save frames
    mag_frames = np.concatenate(mag_frames, 0)
    for i, frame in enumerate(mag_frames):
        fn = os.path.join(save_dir, 'demo_%s_%06d.png'%(args.mode,i))
        im = Image.fromarray(frame)
        im.save(fn) 



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Deep Video Magnification')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--load_ckpt', type=str, metavar='PATH',
                        help='path to load checkpoint')
    parser.add_argument('--save_dir', default='demo', type=str, metavar='PATH',
                        help='path to save generated frames (default: demo)')
    #Device
    parser.add_argument('--device',type=str,default='auto',metavar='',
    help='Select device [auto/cpu/cuda] [Default=auto]')

    parser.add_argument('-m', '--amp', default=20.0, type=float,
                        help='amplification factor (default: 10.0)')
    parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic', 'temporal'],
                        help='amplification mode (static, dynamic, temporal)')
    parser.add_argument('--video_path', default='./../demo_video/baby', type=str, 
                        help='path to video frames')
    parser.add_argument('--num_data', default=300, type=int,
                        help='number of frames')
    #for temporal filter
    parser.add_argument('--fh', default=0.4, type=float)
    parser.add_argument('--fl', default=0.04, type=float)
    #parser.add_argument('--fs', default=30, type=int)
    #parser.add_argument('--ntab', default=2, type=int)

    args = parser.parse_args()

    #Device choice (auto) ======================================================
    if args.device=='auto':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device=args.device

    print(f'Using device: {device}')

    main(args)