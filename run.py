import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image

from utils.data_loader import ImageFromFolderTest
from models.model import VMMpp


def main(args):
    # Create model
    model = VMMpp(img_size=384, patch_size=1, in_chans=3,
                 embed_dim=192, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 manipulator_num_resblk = 1).to(device)

    # Load checkpoint
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
        

    # Check saving directory
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)

    # Data loader
    dataset_mag = ImageFromFolderTest(args.video_path, mag=args.amp, mode=args.mode, num_data=args.num_data, preprocessing=False) 
    data_loader = data.DataLoader(dataset_mag, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.workers,
        pin_memory=False)
    

    # Generate frames
    model.eval()

    # Magnification
    for i, (xa, xb, amp_factor) in enumerate(data_loader):
        if i%10==0: print('processing sample %d'%i)
        amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        xa=xa.to(device)
        xb=xb.to(device)
        amp_factor=amp_factor.to(device)

        y_hat, _, _, _ = model(xa, xb, amp_factor)
        
        if i==0: 
            # Back to image scale (0-255) 
            tmp = xa.permute(0,2,3,1).cpu().detach().numpy()
            tmp = np.clip(tmp, -1.0, 1.0)
            tmp = ((tmp + 1.0) * 127.5).astype(np.uint8)
            
            # Save first frame
            fn = os.path.join(save_dir, 'demo_%s_%06d.png'%(args.mode,i+1))
            im = Image.fromarray(np.concatenate(tmp, 0))
            im.save(fn)
        
        # back to image scale (0-255) 
        y_hat = y_hat.permute(0,2,3,1).cpu().detach().numpy()
        y_hat = np.clip(y_hat, -1.0, 1.0)
        y_hat = ((y_hat + 1.0) * 127.5).astype(np.uint8)
        mag_frames.append(y_hat)

        # Save frames
        fn = os.path.join(save_dir, 'VMM++_%s_%06d.png'%(args.mode,i+1))
        im = Image.fromarray(np.concatenate(mag_frames, 0))
        im.save(fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer Based Video Motion Magnification')
    
    # Compute parameters
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 1)')
    parser.add_argument('--print-freq', '-p', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--load_ckpt', type=str, metavar='PATH',
                        help='path to load checkpoint')
    parser.add_argument('--save_dir', default='demo', type=str, metavar='PATH',
                        help='path to save generated frames (default: demo)')
    
    # Device
    parser.add_argument('--device',type=str,default='auto',metavar='',
    help='Select device [auto/cpu/cuda] [Default=auto]')

    # Application parameters
    parser.add_argument('-m', '--amp', default=20.0, type=float,
                        help='amplification factor (default: 20.0)')
    parser.add_argument('--mode', default='static', type=str, choices=['static', 'dynamic'],
                        help='amplification mode (static, dynamic)')
    parser.add_argument('--video_path', default='./../demo_video/Car_00.mkv', type=str, 
                        help='path to video frames')
    parser.add_argument('--num_data', default=300, type=int,
                        help='number of frames')
    

    args = parser.parse_args()

    # Device choice (auto) ======================================================
    if args.device=='auto':
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device=args.device

    print(f'Using device: {device}')

    main(args)