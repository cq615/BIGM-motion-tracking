import torch.utils.data as data
import numpy as np
import datetime
import os
from os import listdir
import glob
import cv2
import nibabel as nib
from scipy import ndimage
from utils import centre_crop


class Dataset_motion(data.Dataset):
    def __init__(self, data_path, split_set, img_size=96):
        super(Dataset_motion, self).__init__()
        self.data_path = os.path.join(data_path, split_set)
        self.img_size = img_size
        filename = [f.split('_')[0] for f in sorted(listdir(self.data_path))]
        self.filename = list(set(filename))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        # np.random.seed(42)
        # load the nifti images
        disp, mask = load_motion_sim_seq(self.data_path, self.filename[index], self.img_size)
        return disp, mask

    def __len__(self):
        return len(self.filename)


def load_motion_sim_seq(data_path, filename, img_size):
    # Load images and labels, save them into a hdf5 file
    s_num = [f.split('_')[2] for f in glob.glob(os.path.join(data_path, filename+'_slice_*_disp.npy'))]
    slice_n = np.random.choice(s_num)

    disp = np.load(os.path.join(data_path, filename+'_slice_'+slice_n+'_disp.npy'))
    mask = np.load(os.path.join(data_path, filename+'_slice_'+slice_n+'_ED.npy'))

    mask = mask.astype(np.int16)
    kernel = np.ones((3, 3), np.uint8)
    dil_mask = cv2.dilate(mask, kernel, iterations=3)
    dil_mask = np.tile(dil_mask[np.newaxis, np.newaxis], (disp.shape[0], 1, 1, 1))

    disp = centre_crop(disp, size=img_size, centre=[disp.shape[2]//2, disp.shape[3]//2])
    dil_mask = centre_crop(dil_mask, size=img_size, centre=[disp.shape[2] // 2, disp.shape[3] // 2])

    disp = disp / (disp.shape[2] // 2)
    disp= np.transpose(disp, (0, 1, 3, 2))
    disp = np.array(disp, dtype='float32')
    dil_mask = np.transpose(dil_mask, (0, 1, 3, 2))
    dil_mask = np.array(dil_mask, dtype='int16')
    return disp, dil_mask


class Dataset_seq(data.Dataset):
    def __init__(self, data_path):
        super(Dataset_seq, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input_seq, mask = load_data_seq(self.data_path, self.filename[index], size=96)
        image = input_seq[:, :1]
        image_pred = input_seq[:, 1:]
        return image, image_pred, mask[0]

    def __len__(self):
        return len(self.filename)


def load_data_seq(data_path, filename, size):
    nim = nib.load(os.path.join(data_path, filename, 'image_4d.nii.gz'))
    image = nim.get_data()[:, :, :, :]
    res = nim.header['pixdim'][1]

    dim = np.random.randint(1, image.shape[2]-1) # choose one slice
    image_seq = image[:, :, dim]
    image_seq = np.array(image_seq, dtype='float32')

    # preprocessing data
    pl, ph = np.percentile(image_seq, (.01, 99.9))
    image_seq[image_seq < pl] = pl
    image_seq[image_seq > ph] = ph
    image_seq = (image_seq.astype(float) - pl) / (ph - pl)
    new_size = (int(image_seq.shape[1] * res / 1.8), int(image_seq.shape[0] * res / 1.8))
    image_seq = cv2.resize(image_seq, new_size, interpolation=cv2.INTER_LINEAR)
    image_seq = image_seq[np.newaxis].transpose(3, 0, 1, 2)

    image_ed = image_seq[0:1]
    image_seq_bank = np.concatenate((image_seq, np.tile(image_ed,(image_seq.shape[0], 1, 1, 1))), axis=1)

    # load ED segmentation
    nim_seg = nib.load(os.path.join(data_path, filename, 'label_ED.nii.gz'))
    seg = nim_seg.get_data()[:, :, :]
    seg_ed = seg[:, :, dim]
    seg_ed = cv2.resize(seg_ed, new_size, interpolation=cv2.INTER_NEAREST)

    nslice = (seg_ed == 2).astype(np.uint8)
    centre = ndimage.measurements.center_of_mass(nslice)
    centre = np.round(centre).astype(np.uint8)

    image_seq_bank = centre_crop(image_seq_bank, size, centre)
    image_seq_bank = np.transpose(image_seq_bank, (0, 1, 3, 2)).astype('float32')

    # create dilated mask 
    mask = (seg_ed == 2).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    mask = mask[np.newaxis, np.newaxis]
    mask = centre_crop(mask, size, centre)
    mask = np.array(mask, dtype='int16')
    return image_seq_bank, mask
