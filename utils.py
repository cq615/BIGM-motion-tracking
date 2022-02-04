from scipy import signal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def transform(seg_source, loc, mode='bilinear'):
    grid = generate_grid(seg_source, loc)
    out = F.grid_sample(seg_source, grid, mode=mode)
    return out

def generate_grid(x, offset):
    x_shape = x.size()
    grid_w, grid_h = torch.meshgrid([torch.linspace(-1, 1, x_shape[2]), torch.linspace(-1, 1, x_shape[3])])  # (h, w)
    grid_w = grid_w.cuda().float()
    grid_h = grid_h.cuda().float()

    grid_w = nn.Parameter(grid_w, requires_grad=False)
    grid_h = nn.Parameter(grid_h, requires_grad=False)

    offset_h, offset_w = torch.split(offset, 1, 1)
    offset_w = offset_w.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)
    offset_h = offset_h.contiguous().view(-1, int(x_shape[2]), int(x_shape[3]))  # (b*c, h, w)

    offset_w = grid_w + offset_w
    offset_h = grid_h + offset_h

    offsets = torch.stack((offset_h, offset_w), 3)
    return offsets

def define_weight(length=60, std=10):
    window = signal.gaussian(2*length, std=std)
    window1 = window[:length]
    window2 = window[length:]
    weight = np.concatenate((window1[::3], window2[::2]), axis=0)
    weight = weight/np.sum(weight)
    return weight

def compute_gradient(x):
    bsize, csize, height, width = x.size()
    u = torch.cat((torch.zeros(bsize, csize, height, 1).cuda(), x, torch.zeros(bsize, csize, height, 1).cuda()), 3)
    d_x = (torch.index_select(u, 3, torch.arange(2, width+2).cuda()) - torch.index_select(u, 3, torch.arange(width).cuda()))/2
    v = torch.cat((torch.zeros(bsize, csize, 1, width).cuda(), x, torch.zeros(bsize, csize, 1, width).cuda()), 2)
    d_y = (torch.index_select(v, 2, torch.arange(2, height+2).cuda()) - torch.index_select(v, 2, torch.arange(height).cuda()))/2
    d_xy = torch.cat((d_x, d_y), 1)
    d_xy = torch.index_select(d_xy, 1, torch.tensor([0, 2, 1, 3]).cuda())
    return d_xy

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight*torch.sum((input - target) ** 2, (3, 2, 1)))

def MotionVAELoss_weighted(recon_x, x, weight, mu, logvar, beta=1e-2):
    BCE = weighted_mse_loss(recon_x, x, weight)
    KLD = -0.5 * torch.mean(weight*torch.sum((1 + logvar - mu.pow(2) - logvar.exp()), 1))
    return BCE + beta*KLD

def centre_crop(img, size, centre):
    img_new = np.zeros((img.shape[0],img.shape[1],size,size))
    h1 = np.amin([size//2, centre[0]])
    h2 = np.amin([size//2, img.shape[2]-centre[0]])
    w1 = np.amin([size//2, centre[1]])
    w2 = np.amin([size//2, img.shape[3]-centre[1]])
    img_new[:,:,size//2-h1:size//2+h2,size//2-w1:size//2+w2] = img[:,:,centre[0]-h1:centre[0]+h2,centre[1]-w1:centre[1]+w2]
    return img_new