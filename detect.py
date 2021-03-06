# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch

import config
from models.decoder import Decoder
from models.encoder import Encoder
from util.common import num_channels, load_checkpoint, plot_volumes, to_volume
from util.dataset import LoadImages


def detect(path, encoder=None, decoder=None):
    torch.backends.cudnn.benchmark = True

    dataset = LoadImages(path, img_size=config.IMAGE_SIZE, used_layers=config.USED_LAYERS)

    if not encoder or not decoder:
        in_channels = num_channels(config.USED_LAYERS)
        encoder = Encoder(in_channels=in_channels)
        decoder = Decoder(num_classes=config.NUM_CLASSES+1)
        encoder = encoder.to(config.DEVICE)
        decoder = decoder.to(config.DEVICE)

        _, encoder, decoder = load_checkpoint(encoder, decoder, config.CHECKPOINT_FILE, config.DEVICE)

    encoder.eval()
    decoder.eval()

    for _, layers, path in dataset:
        with torch.no_grad():
            layers = torch.from_numpy(layers).to(config.DEVICE, non_blocking=True)
            if layers.ndimension() == 3:
                layers = layers.unsqueeze(0)

            features = encoder(layers)
            predictions = decoder(features)
            _, out = predictions, predictions.sigmoid()

            plot_volumes(to_volume(out, config.VOXEL_THRESH).cpu(), [path], config.NAMES)


if __name__ == "__main__":
    detect(config.DETECT_PATH)
