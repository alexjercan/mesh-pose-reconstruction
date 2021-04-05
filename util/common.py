# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import glob
import os
from pathlib import Path

import cv2
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import torch

L_RGB = 'rgb'
L_DEPTH = 'depth'
L_NORMAL = 'normal'

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo', 'exr']  # acceptable image suffixes


def num_channels(layers):
    # Get the number of channels based on a layer format
    # e.g (L_RGB, L_DEPTH, L_NORMAL)
    nc = {L_RGB: 3, L_DEPTH: 3, L_NORMAL: 3}
    return sum([nc[k] for k in layers])


def resize_img(img, img_size=False, augment=False):
    h0, w0 = img.shape[:2]  # orig hw
    r = 1 if not img_size else img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def exr2depth(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    # get the maximum value from the array, aka the most distant point
    # everything above that value is infinite, thus i clamp it to maxvalue
    # then divide by maxvalue to obtain a normalized map
    # multiply by 255 to obtain a colormap from the depth map
    maxvalue = np.max(img[img < np.max(img)])
    img[img > maxvalue] = maxvalue
    img = img / maxvalue * 255

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.array(img).astype(np.uint8).reshape((img.shape[0], img.shape[1], -1))

    return img


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    img[img > 1] = 1
    img[img < 0] = 0
    img = img * 255

    img = np.array(img).astype(np.uint8).reshape((img.shape[0], img.shape[1], -1))

    return img


def pkl2mesh(path):
    with open(path, 'rb') as f:
        meshes = pickle.load(f)
    return meshes  # [class, vertices, edges]xN


def img2bgr(path):
    return cv2.imread(path)  # BGR


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def load_mesh_paths(path: str):
    path = Path(path)
    f = glob.glob(str(path / '**' / '*mesh.pkl'), recursive=True)
    label_files = sorted([x.replace('/', os.sep) for x in f])
    return label_files


def load_img_paths(path: str, used_layers):
    layer_files = {}
    path = Path(path)
    f = glob.glob(str(path / '**' / '*.*'), recursive=True)
    for layer in used_layers:
        layer_files[layer] = sorted([x.replace('/', os.sep) for x in f if
                                     x.split('.')[-1].lower() in img_formats and layer.lower() in x.lower()])
    return layer_files


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def load_checkpoint(encoder, decoder, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    init_epoch = checkpoint['epoch_idx']
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    return init_epoch, encoder, decoder


def save_checkpoint(epoch_idx, encoder, decoder, dir_checkpoints):
    file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx + 1)
    output_path = os.path.join(dir_checkpoints, file_name)
    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)
    checkpoint = {
        'epoch_idx': epoch_idx,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }
    torch.save(checkpoint, output_path)


def plot_volumes(volumes, img_files):
    if isinstance(volumes, torch.Tensor):
        volumes = volumes.numpy()
    cmap = cm.get_cmap('hsv', 100)
    for volume, img_file in zip(volumes, img_files):
        voxels = volume > 0
        colors = np.zeros(voxels.shape + (4,))
        colors[..., 0:4] = cmap(volume/100)
        ax = plt.axes(projection='3d')
        ax.voxels(voxels, facecolors=colors,
                  edgecolors=np.clip(2*colors - 0.5, 0, 1),
                  linewidth=0.5)
        ax.set_title(img_file)
        plt.show()


def predictions_iou(pr_volumes, gt_volumes):
    gt_volumes = torch.ge(gt_volumes, 1).float()
    pr_volumes = torch.ge(pr_volumes, 1).float()
    intersection = torch.sum(pr_volumes.mul(gt_volumes), dim=(1,2,3)).float()
    union = torch.sum(torch.ge(pr_volumes.add(gt_volumes), 1), dim=(1,2,3)).float()
    return (intersection / union).mean().item()


def to_volume(predictions, threshold=0.5):
    conf, pred = torch.max(predictions, dim=1)
    pred = torch.where(conf > threshold, pred, torch.zeros_like(pred))
    return pred
