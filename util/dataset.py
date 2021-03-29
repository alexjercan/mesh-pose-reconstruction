# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py

import os

import cv2
import torch
import numpy as np

from copy import copy
from torch.utils.data import Dataset, DataLoader

from util.common import exr2normal, exr2depth, img2bgr, pkl2mesh, resize_img, load_mesh_paths, img_formats, L_RGB, \
    L_DEPTH, L_NORMAL, load_img_paths, plot_volumes


def create_dataloader(img_path, mesh_path, batch_size=2, used_layers=None, img_size=224, map_size=32, augment=False,
                      workers=8, pin_memory=True):
    dataset = BDataset(img_path, mesh_path, used_layers=used_layers, img_size=img_size, map_size=map_size,
                       augment=augment)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory)
    return dataset, dataloader


class BDataset(Dataset):
    def __init__(self, img_path, mesh_path, used_layers=None, img_size=224, map_size=32, augment=False):
        super(BDataset, self).__init__()
        if used_layers is None:
            used_layers = [L_RGB]
        self.img_path = img_path
        self.mesh_path = mesh_path
        self.used_layers = used_layers
        self.img_size = img_size
        self.map_size = map_size
        self.augment = augment

        self.layer_files = load_img_paths(self.img_path, used_layers)
        self.img_files = self.layer_files[self.used_layers[0]]
        self.mesh_files = load_mesh_paths(self.mesh_path)
        self.layers = [None] * len(self.img_files)
        self.meshes = [None] * len(self.img_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        layers, voxels, (h0, w0), (h, w) = load_data(self, index)
        layers = {k: letterbox(layers[k], self.img_size, auto=False, scale_up=self.augment)[0] for k in layers}

        # Convert
        if L_RGB in layers:
            layers[L_RGB] = layers[L_RGB][:, :, ::-1]
        layers = {k: layers[k].transpose(2, 0, 1) for k in layers}
        img0 = layers[self.used_layers[0]]
        layers = [layers[k] for k in self.used_layers]
        layers = np.concatenate(layers, axis=0).astype(np.float32) / 255.0

        img0 = np.ascontiguousarray(img0)
        layers = np.ascontiguousarray(layers)

        return torch.from_numpy(img0), torch.from_numpy(layers), torch.from_numpy(voxels)


def load_data(self, index):
    layers0 = self.layers[index]
    if layers0 is not None:
        return copy(layers0), self.img_hw0[index], self.img_hw[index]

    layers0 = {}
    hw0, hw = (0, 0), (0, 0)
    if L_RGB in self.used_layers:
        img0, hw0, hw = load_image(self.layer_files[L_RGB], index, self.img_size, self.augment)
        layers0[L_RGB] = img0
    if L_DEPTH in self.used_layers:
        depth0, hw0, hw = load_depth(self.layer_files[L_DEPTH], index, self.img_size, self.augment)
        layers0[L_DEPTH] = depth0
    if L_NORMAL in self.used_layers:
        normal0, hw0, hw = load_normal(self.layer_files[L_NORMAL], index, self.img_size, self.augment)
        layers0[L_NORMAL] = normal0
    voxels0 = load_voxels(self.mesh_files, index, self.map_size)

    return layers0, voxels0, hw0, hw


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scale_fill=False, scale_up=True, stride=32):
    # Borrowed from https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_pad[0], new_shape[0] - new_pad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_pad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_pad:  # resize
        img = cv2.resize(img, new_pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def load_image(layer_files, index, img_size, augment=None):
    path = layer_files[index]
    img = img2bgr(path)  # BGR
    assert img is not None, 'Image Not Found ' + path
    return resize_img(img, img_size, augment)


def load_depth(layer_files, index, img_size, augment=None):
    path = layer_files[index]
    img = exr2depth(path)  # 3 channel depth
    assert img is not None, 'Image Not Found ' + path
    return resize_img(img, img_size, augment)


def load_normal(layer_files, index, img_size, augment=None):
    path = layer_files[index]
    img = exr2normal(path)  # 3 channel normal
    assert img is not None, 'Image Not Found ' + path
    return resize_img(img, img_size, augment)


def load_voxels(mesh_files, index, map_size):
    meshes = pkl2mesh((mesh_files[index]))
    voxels = np.zeros((map_size, map_size, map_size)).astype(np.float32)

    vertices = np.concatenate([vs for (_, vs, _) in meshes], axis=1)
    vertices[:, 0] *= (map_size - 1) / np.max(vertices[:, 0])
    vertices[:, 1] *= (map_size - 1) / np.max(vertices[:, 1])
    vertices[:, 2] *= (map_size - 1) / np.max(vertices[:, 2])
    vertices = np.floor(vertices).astype(dtype=np.int32)
    vertices = np.unique(vertices, axis=0)

    voxels[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1
    return voxels


if __name__ == "__main__":
    ds, dl = create_dataloader("../../bdataset/images/train", "../../bdataset/labels/train", batch_size=10)
    i0, ls, vxs = next(iter(dl))
    plot_volumes(vxs)
