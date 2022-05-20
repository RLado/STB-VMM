import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import VMMpp
from data_loader import ImageFromFolder
from utils import AverageMeter

parser = argparse.ArgumentParser(description='PyTorch Deep Video Magnification')
parser.add_argument('-d', '--dataset', default='./../data/train', type=str, 
                    help='Path to the train folder of the dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=12, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--num_data', default=100000, type=int,
                    help='number of total data sample used for training (default: 100000)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--ckpt', default='ckpt', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ckpt)')
#Device
parser.add_argument('--device',type=str,default='auto',metavar='',
help='Select device [auto/cpu/cuda] [Default=auto]')
parser.add_argument('--weight_reg1', default=1.0, type=float,
                    help='weight texture regularization loss  (default: 1.0)')
args = parser.parse_args()


#Device choice (auto) ======================================================
if args.device=='auto':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device=args.device

losses_recon, losses_reg1 = [],[]
def main():
    global args
    global losses_recon, losses_reg1
    args = parser.parse_args()
    print(args)

    # create model
    model = VMMpp(img_size=384, patch_size=1, in_chans=3,
                 embed_dim=48, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, img_range=1., resi_connection='1conv',
                 manipulator_num_resblk = 1).to(device)
    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            model.load_state_dict(checkpoint['state_dict'])
            losses_recon = checkpoint['losses_recon'] 
            losses_reg1 = checkpoint['losses_reg1']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # check saving directory
    ckpt_dir = args.ckpt
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    print(ckpt_dir)

    # dataloader
    dataset_mag = ImageFromFolder(args.dataset, num_data=args.num_data, preprocessing=True) 
    data_loader = data.DataLoader(dataset_mag, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers,
        pin_memory=False)

    # loss criterion
    criterion = nn.L1Loss(reduction='mean').to(device)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                betas=(0.9,0.999),
                                weight_decay=args.weight_decay)


    # train model
    for epoch in range(args.start_epoch, args.epochs):
        loss_recon, loss_reg1 = train(data_loader, model, criterion, optimizer, epoch, args)
        
        # stack losses
        losses_recon.append(loss_recon)
        losses_reg1.append(loss_reg1)

        dict_checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.to('cpu').state_dict(), #pass model to cpu to avoid problems at load time
            'losses_recon': losses_recon,
            'losses_reg1': losses_reg1
        }
        model.to(device) #Return model to device

        # save checkpoints
        fpath = os.path.join(ckpt_dir, 'ckpt_e%02d.pth.tar'%(epoch))
        torch.save(dict_checkpoint, fpath)

def train(loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_recon = AverageMeter()
    losses_reg1 = AverageMeter() # texture loss

    model.train()

    end = time.time()
    for i, (y, xa, xb, xc, amp_factor) in enumerate(loader):
        y = y.to(device)
        xa = xa.to(device)
        xb = xb.to(device)
        xc = xc.to(device)
        amp_factor = amp_factor.to(device)
        data_time.update(time.time() - end)

        # compute output
        amp_factor = amp_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        y_hat, rep_a, rep_b = model(xa, xb, amp_factor)

        # compute losses
        loss_recon = criterion(y_hat, y)
        loss_reg1 = args.weight_reg1 * L1_loss(rep_a, rep_b)
        loss = loss_recon + loss_reg1

        losses_recon.update(loss_recon.item()) 
        losses_reg1.update(loss_reg1.item()) 

        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'LossR1 {loss_reg1.val:.4f} ({loss_reg1.avg:.4f})\t'.format(
                   epoch, i, len(loader), batch_time=batch_time, data_time=data_time, 
                   loss=losses_recon, loss_reg1=losses_reg1))
    

    return losses_recon.avg, losses_reg1.avg


def L1_loss(input, target):
    return torch.abs(input - target).mean()


 
if __name__ == '__main__':
    main()
