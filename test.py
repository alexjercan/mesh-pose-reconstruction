# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
from tqdm import tqdm

import config
from models.decoder import Decoder
from models.encoder import Encoder
from util.common import num_channels, load_checkpoint, var_or_cuda, plot_volumes
from util.dataset import create_dataloader


def test(encoder=None, decoder=None):
    torch.backends.cudnn.benchmark = True

    dataset, dataloader = create_dataloader(config.IMG_DIR + "/test", config.MESH_DIR + "/test",
                                            batch_size=config.BATCH_SIZE, used_layers=config.USED_LAYERS,
                                            img_size=config.IMAGE_SIZE, map_size=config.MAP_SIZE,
                                            augment=config.AUGMENT, workers=config.NUM_WORKERS,
                                            pin_memory=config.PIN_MEMORY)
    if not encoder or not decoder:
        in_channels = num_channels(config.USED_LAYERS)
        encoder = Encoder(in_channels=in_channels)
        decoder = Decoder()
        encoder = encoder.to(config.DEVICE)
        decoder = decoder.to(config.DEVICE)

        epoch_idx, encoder, decoder = load_checkpoint(encoder, decoder, config.CHECKPOINT_FILE, config.DEVICE)

    bce_loss = torch.nn.BCELoss()

    loop = tqdm(dataloader, leave=True)
    encoder_losses = []

    encoder.eval()
    decoder.eval()

    for i, (img0s, layers, volumes) in enumerate(loop):
        with torch.no_grad():
            layers = var_or_cuda(layers)
            volumes = var_or_cuda(volumes)

            features = encoder(layers)
            predictions = decoder(features)
            predictions = torch.mean(predictions, dim=1)
            encoder_loss = bce_loss(predictions, volumes) * 10

            encoder_losses.append(encoder_loss.item())

            sample_iou = []
            for th in config.VOXEL_THRESH:
                _volume = torch.ge(predictions, th).float()
                intersection = torch.sum(_volume.mul(volumes)).float()
                union = torch.sum(torch.ge(_volume.add(volumes), 1)).float()
                sample_iou.append((intersection / union).item())

            mean_loss = sum(encoder_losses) / len(encoder_losses)
            sample_iou = ['%.4f' % si for si in sample_iou]
            loop.set_postfix(loss=mean_loss, iou=sample_iou)

            if i == 0 and config.PLOT:
                plot_volumes(predictions.cpu(), th=0.5)


if __name__ == "__main__":
    test()
