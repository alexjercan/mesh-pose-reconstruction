# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import re

import config
import torch
from tqdm import tqdm

from datetime import datetime as dt
from models.decoder import Decoder
from models.encoder import Encoder
from models.loss import LossFunction
from test import test
from util.common import num_channels, init_weights, save_checkpoint, load_checkpoint
from util.dataset import create_dataloader


def train_one_epoch(encoder, decoder, dataloader, loss_fn, encoder_solver, decoder_solver):
    loop = tqdm(dataloader, leave=True)
    losses = []

    for _, (_, layers, volumes, _) in enumerate(loop):
        layers = layers.to(config.DEVICE, non_blocking=True)
        volumes = volumes.to(config.DEVICE, non_blocking=True)

        features = encoder(layers)
        predictions = decoder(features)
        loss = loss_fn(predictions, volumes)

        losses.append(loss.item())

        encoder.zero_grad()
        decoder.zero_grad()
        loss.backward()
        encoder_solver.step()
        decoder_solver.step()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def train():
    torch.backends.cudnn.benchmark = True

    _, dataloader = create_dataloader(config.IMG_DIR + "/train", config.MESH_DIR + "/train",
                                            batch_size=config.BATCH_SIZE, used_layers=config.USED_LAYERS,
                                            img_size=config.IMAGE_SIZE, map_size=config.MAP_SIZE,
                                            augment=config.AUGMENT, workers=config.NUM_WORKERS,
                                            pin_memory=config.PIN_MEMORY, shuffle=True)

    in_channels = num_channels(config.USED_LAYERS)
    encoder = Encoder(in_channels=in_channels)
    decoder = Decoder(num_classes=config.NUM_CLASSES+1)
    encoder.apply(init_weights)
    decoder.apply(init_weights)
    encoder_solver = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()),
                                      lr=config.ENCODER_LEARNING_RATE,
                                      betas=config.BETAS)
    decoder_solver = torch.optim.Adam(decoder.parameters(),
                                      lr=config.DECODER_LEARNING_RATE,
                                      betas=config.BETAS)
    encoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_solver,
                                                                milestones=config.ENCODER_LR_MILESTONES,
                                                                gamma=config.GAMMA)
    decoder_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(decoder_solver,
                                                                milestones=config.DECODER_LR_MILESTONES,
                                                                gamma=config.GAMMA)
    encoder = encoder.to(config.DEVICE)
    decoder = decoder.to(config.DEVICE)

    loss_fn = LossFunction()

    init_epoch = 0
    if config.CHECKPOINT_FILE and config.LOAD_MODEL:
        init_epoch, encoder, decoder = load_checkpoint(encoder, decoder, config.CHECKPOINT_FILE, config.DEVICE)

    output_dir = os.path.join(config.OUT_PATH, '%s', re.sub("[^0-9a-zA-Z]+", "-", dt.now().isoformat()))
    runs_dir = output_dir % 'runs'

    for epoch_idx in range(init_epoch, config.NUM_EPOCHS):
        encoder.train()
        decoder.train()
        train_one_epoch(encoder, decoder, dataloader, loss_fn, encoder_solver, decoder_solver)
        encoder_lr_scheduler.step()
        decoder_lr_scheduler.step()

        if config.TEST:
            test(encoder, decoder)
        if config.SAVE_MODEL:
            save_checkpoint(epoch_idx, encoder, decoder, runs_dir)

    if not config.TEST:
        test(encoder, decoder)
    if not config.SAVE_MODEL:
        save_checkpoint(config.NUM_EPOCHS - 1, encoder, decoder, runs_dir)


if __name__ == "__main__":
    train()
