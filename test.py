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
from models.loss import LossFunction
from util.common import num_channels, load_checkpoint, plot_volumes, predictions_iou, to_volume
from util.dataset import create_dataloader


def test(encoder=None, decoder=None):
    torch.backends.cudnn.benchmark = True

    _, dataloader = create_dataloader(config.IMG_DIR + "/test", config.MESH_DIR + "/test",
                                            batch_size=config.BATCH_SIZE, used_layers=config.USED_LAYERS,
                                            img_size=config.IMAGE_SIZE, map_size=config.MAP_SIZE,
                                            augment=config.AUGMENT, workers=config.NUM_WORKERS,
                                            pin_memory=config.PIN_MEMORY, shuffle=False)
    if not encoder or not decoder:
        in_channels = num_channels(config.USED_LAYERS)
        encoder = Encoder(in_channels=in_channels)
        decoder = Decoder(num_classes=config.NUM_CLASSES+1)
        encoder = encoder.to(config.DEVICE)
        decoder = decoder.to(config.DEVICE)

        _, encoder, decoder = load_checkpoint(encoder, decoder, config.CHECKPOINT_FILE, config.DEVICE)

    loss_fn = LossFunction()

    loop = tqdm(dataloader, leave=True)
    losses = []
    ious = []

    encoder.eval()
    decoder.eval()

    for i, (_, layers, volumes, img_files) in enumerate(loop):
        with torch.no_grad():
            layers = layers.to(config.DEVICE, non_blocking=True)
            volumes = volumes.to(config.DEVICE, non_blocking=True)

            features = encoder(layers)
            predictions = decoder(features)

            loss = loss_fn(predictions, volumes)
            losses.append(loss.item())

            iou = predictions_iou(to_volume(predictions, config.VOXEL_THRESH), volumes)
            ious.append(iou)

            mean_iou = sum(ious) / len(ious)
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss=mean_loss, mean_iou=mean_iou)

            if i == 0 and config.PLOT:
                plot_volumes(to_volume(predictions, config.VOXEL_THRESH).cpu(), img_files, config.NAMES)
                plot_volumes(volumes.cpu(), img_files, config.NAMES)


if __name__ == "__main__":
    test()
