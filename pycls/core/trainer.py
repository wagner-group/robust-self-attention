#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tools for training and testing a model."""

import random

from contextlib import nullcontext
import numpy as np
import pycls.core.attacks as attacks
import pycls.core.benchmark as benchmark
import pycls.core.builders as builders
import pycls.core.checkpoint as cp
import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.logging as logging
import pycls.core.meters as meters
import pycls.core.net as net
import pycls.core.optimizer as optim
import pycls.datasets.loader as data_loader
import timm
import torch
import torch.cuda.amp as amp
import torchvision.transforms.functional as TF
from pycls.core.config import cfg
from pycls.core.io import pathmgr
import wandb


logger = logging.get_logger(__name__)


def setup_env():
    """Sets up environment for training or testing."""
    if dist.is_master_proc():
        # Ensure that the output dir exists
        pathmgr.mkdirs(cfg.OUT_DIR)
        # Save the config
        config.dump_cfg()
        if cfg.WANDB.ENABLED:
            wandb.init(entity=cfg.WANDB.ENTITY, project=cfg.WANDB.PROJECT,
                group=cfg.WANDB.GROUP, id=cfg.WANDB.RUN_ID, config=config.to_dict(cfg))
    # Setup logging
    logging.setup_logging()
    # Log torch, cuda, and cudnn versions
    version = [torch.__version__, torch.version.cuda, torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    # Log the config as both human readable and as a json
    logger.info("Config:\n{}".format(cfg)) if cfg.VERBOSE else ()
    logger.info(logging.dump_log_data(cfg, "cfg", None))
    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK


def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    if cfg.MODEL.TIMM_MODEL:
        model = timm.models.create_model(cfg.MODEL.TIMM_MODEL,
            num_classes=cfg.MODEL.NUM_CLASSES,
            pretrained=cfg.MODEL.TIMM_PRETRAINED)
    else:
        model = builders.build_model()
    logger.info("Model:\n{}".format(model)) if cfg.VERBOSE else ()
    # Log model complexity
    if hasattr(model, 'complexity'):
        logger.info(logging.dump_log_data(net.complexity(model), "complexity"))
    # Transfer the model to the current GPU device
    err_str = "Cannot use more GPU devices than available"
    assert cfg.NUM_GPUS <= torch.cuda.device_count(), err_str
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        ddp = torch.nn.parallel.DistributedDataParallel
        model = ddp(module=model, device_ids=[cur_device], output_device=cur_device,
                    find_unused_parameters=len(cfg.OPTIM.TRAINABLE_PARAMS) > 0)
    return model


class NormalizeLayer(torch.nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
        # Per-channel mean, std always specified in RGB order
        if cfg.DATA_LOADER.BGR:
            self.mean = cfg.DATA_LOADER.MEAN[::-1]
            self.std = cfg.DATA_LOADER.STD[::-1]
        else:
            self.mean = cfg.DATA_LOADER.MEAN
            self.std = cfg.DATA_LOADER.STD

    def forward(self, x):
        return TF.normalize(x, mean=self.mean, std=self.std)


class DenormalizeLayer(torch.nn.Module):
    def __init__(self):
        super(DenormalizeLayer, self).__init__()
        # Per-channel mean, std always specified in RGB order
        if cfg.DATA_LOADER.BGR:
            mean = cfg.DATA_LOADER.MEAN[::-1]
            std = cfg.DATA_LOADER.STD[::-1]
        else:
            mean = cfg.DATA_LOADER.MEAN
            std = cfg.DATA_LOADER.STD

        self.mean = [-mean[0], -mean[1], -mean[2]]
        self.std = [1/std[0], 1/std[1], 1/std[2]]

    def forward(self, x):
        x = TF.normalize(x, mean=[0, 0, 0], std=self.std)
        x = TF.normalize(x, mean=self.mean, std=[1, 1, 1])
        return x


def train_epoch(loader, model, loss_fun, optimizer, scaler, meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Convert labels to smoothed one-hot vector
        labels_one_hot = net.smooth_one_hot_labels(labels)
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        # Perform the forward pass and compute the loss
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            preds = model(inputs)
            loss = loss_fun(preds, labels_one_hot)
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
        if dist.is_master_proc() and cfg.WANDB.ENABLED:
            wandb.log({'train_acc1': 100 - top1_err,
                       'train_acc5': 100 - top5_err,
                       'train_loss': loss, 'lr': lr})
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


def train_adv_epoch(loader, model, loss_fun, optimizer, scaler, meter, cur_epoch):
    """Performs one epoch of training."""
    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    model_wrapped = torch.nn.Sequential(NormalizeLayer(), model)
    denormalize = DenormalizeLayer()
    meter.reset()
    meter.iter_tic()
    if cfg.ADV.TRAIN_ATTACK == "patchpgd":
        adversary = attacks.PatchPGDTrain(model_wrapped,
                                          patch_size=cfg.ADV.TRAIN_PATCH_SIZES,
                                          img_size=cfg.TRAIN.IM_SIZE,
                                          num_steps=cfg.ADV.TRAIN_NUM_STEPS,
                                          step_size=cfg.ADV.TRAIN_STEP_SIZE,
                                          grid_aligned=cfg.ADV.TRAIN_GRID_ALIGNED).cuda()
    else:
        raise Exception('Unknown attack name: {}'.format(cfg.ADV.TRAIN_ATTACK))

    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Denormalize inputs for ease of clipping
        inputs = denormalize(inputs)
        # Convert labels to smoothed one-hot vector
        labels_one_hot = net.smooth_one_hot_labels(labels)
        # Apply mixup to the batch (no effect if mixup alpha is 0)
        inputs, labels_one_hot, labels = net.mixup(inputs, labels_one_hot)
        # Perform the forward pass and compute the loss
        with amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Adversarially perturb batch
            inputs_p = adversary(inputs, labels)
            preds = model_wrapped(inputs_p)
            loss = loss_fun(preds, labels_one_hot)
        # Perform the backward pass and update the parameters
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs (no reduction if 1 GPU used)
        loss, top1_err, top5_err = dist.scaled_all_reduce([loss, top1_err, top5_err])
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err, top5_err = loss.item(), top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        mb_size = inputs.size(0) * cfg.NUM_GPUS
        meter.update_stats(top1_err, top5_err, loss, lr, mb_size)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
        if dist.is_master_proc() and cfg.WANDB.ENABLED:
            wandb.log({'train_adv_acc1': 100 - top1_err,
                       'train_adv_acc5': 100 - top5_err,
                       'train_adv_loss': loss, 'lr': lr})
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)


@torch.no_grad()
def test_epoch(loader, model, meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    meter.reset()
    meter.iter_tic()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        with amp.autocast(enabled=cfg.TEST.MIXED_PRECISION):
            preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)
    if dist.is_master_proc() and cfg.WANDB.ENABLED:
        stats = meter.get_epoch_stats(cur_epoch)
        wandb.log({'test_acc1': 100 - stats['top1_err'],
                   'test_acc5': 100 - stats['top5_err']})


@torch.no_grad()
def test_adv_epoch(loader, model, meter, cur_epoch):
    """Evaluates the model on the test set."""
    # Enable eval mode
    model.eval()
    # Wrap model with an input normalization layer
    model_wrapped = torch.nn.Sequential(NormalizeLayer(), model)
    denormalize = DenormalizeLayer()
    meter.reset()
    meter.iter_tic()
    if cfg.ADV.VAL_ATTACK == 'patchpgd':
        adversary = attacks.PatchPGD(model_wrapped,
                                     patch_size=cfg.ADV.VAL_PATCH_SIZE,
                                     img_size=cfg.TEST.IM_SIZE,
                                     num_steps=cfg.ADV.VAL_NUM_STEPS,
                                     step_size=cfg.ADV.VAL_STEP_SIZE,
                                     num_restarts=cfg.ADV.VAL_NUM_RESTARTS,
                                     grid_aligned=cfg.ADV.VAL_GRID_ALIGNED,
                                     verbose=cfg.ADV.VAL_VERBOSE).cuda()
    elif cfg.ADV.VAL_ATTACK == 'patchpgdgrid':
        adversary = attacks.PatchPGDGrid(model_wrapped,
                                         patch_size=cfg.ADV.VAL_PATCH_SIZE,
                                         patch_stride=cfg.ADV.VAL_PATCH_STRIDE,
                                         img_size=cfg.TEST.IM_SIZE,
                                         num_steps=cfg.ADV.VAL_NUM_STEPS,
                                         step_size=cfg.ADV.VAL_STEP_SIZE,
                                         verbose=cfg.ADV.VAL_VERBOSE).cuda()
    elif cfg.ADV.VAL_ATTACK == 'patchautopgd':
        adversary = attacks.PatchAutoPGD(model_wrapped,
                                         patch_size=cfg.ADV.VAL_PATCH_SIZE,
                                         img_size=cfg.TEST.IM_SIZE,
                                         num_steps=cfg.ADV.VAL_NUM_STEPS,
                                         num_restarts=cfg.ADV.VAL_NUM_RESTARTS,
                                         grid_aligned=cfg.ADV.VAL_GRID_ALIGNED,
                                         verbose=cfg.ADV.VAL_VERBOSE).cuda()
    elif cfg.ADV.VAL_ATTACK == 'patchautopgdgrid':
        adversary = attacks.PatchAutoPGDGrid(model_wrapped,
                                             patch_size=cfg.ADV.VAL_PATCH_SIZE,
                                             patch_stride=cfg.ADV.VAL_PATCH_STRIDE,
                                             img_size=cfg.TEST.IM_SIZE,
                                             num_steps=cfg.ADV.VAL_NUM_STEPS,
                                             verbose=cfg.ADV.VAL_VERBOSE).cuda()
    else:
        raise Exception('Unknown attack name: {}'.format(cfg.ADV.VAL_ATTACK))

    for cur_iter, (inputs, labels) in enumerate(loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Denormalize inputs for ease of clipping
        inputs = denormalize(inputs)
        # Disable DDP synchronization
        with model.no_sync() if cfg.NUM_GPUS > 1 else nullcontext():
            with amp.autocast(enabled=cfg.TEST.MIXED_PRECISION):
                # Perturb inputs
                inputs = adversary(inputs, labels)
                # Compute the predictions
                preds = model_wrapped(inputs)
        # Compute the errors
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs  (no reduction if 1 GPU used)
        top1_err, top5_err = dist.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.iter_toc()
        # Update and log stats
        meter.update_stats(top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS)
        meter.log_iter_stats(cur_epoch, cur_iter)
        meter.iter_tic()
    # Log epoch stats
    meter.log_epoch_stats(cur_epoch)
    if dist.is_master_proc() and cfg.WANDB.ENABLED:
        stats = meter.get_epoch_stats(cur_epoch)
        wandb.log({'test_adv_acc1': 100 - stats['top1_err'],
                   'test_adv_acc5': 100 - stats['top5_err']})


def train_model():
    """Trains the model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model, loss_fun, and optimizer
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    optimizer = optim.construct_optimizer(model)
    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cp.has_checkpoint():
        file = cp.get_last_checkpoint()
        epoch = cp.load_checkpoint(file, model, optimizer)
        logger.info("Loaded checkpoint from: {}".format(file))
        start_epoch = epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        cp.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info("Loaded initial weights from: {}".format(cfg.TRAIN.WEIGHTS))
    # Create data loaders and meters
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    train_meter = meters.TrainMeter(len(train_loader))
    test_meter = meters.TestMeter(len(test_loader))
    if cfg.ADV.VAL_ATTACK:
        test_adv_meter = meters.TestMeter(len(test_loader))
    # Create a GradScaler for mixed precision training
    scaler = amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    # Compute model and loader timings
    if start_epoch == 0 and cfg.PREC_TIME.NUM_ITER > 0:
        benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
    # Perform the training loop
    logger.info("Start epoch: {}".format(start_epoch + 1))
    best_err = np.inf

    if start_epoch >= cfg.OPTIM.MAX_EPOCH:
        return

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        params = (train_loader, model, loss_fun, optimizer, scaler, train_meter)
        if cfg.ADV.TRAIN_ATTACK:
            train_adv_epoch(*params, cur_epoch)
        else:
            train_epoch(*params, cur_epoch)
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            net.compute_precise_bn_stats(model, train_loader)
        # Evaluate the model
        test_epoch(test_loader, model, test_meter, cur_epoch)
        if cfg.ADV.VAL_ATTACK:
            test_adv_epoch(test_loader, model, test_adv_meter, cur_epoch)
        # Check if checkpoint is best so far (note: should checkpoint meters as well)
        stats = test_meter.get_epoch_stats(cur_epoch)
        best = stats["top1_err"] <= best_err
        best_err = min(stats["top1_err"], best_err)
        # Save a checkpoint
        file = cp.save_checkpoint(model, optimizer, cur_epoch, best)
        logger.info("Wrote checkpoint to: {}".format(file))


def test_model():
    """Evaluates a trained model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model
    model = setup_model()
    # Load model weights
    if cfg.TEST.WEIGHTS:
        cp.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info("Loaded model weights from: {}".format(cfg.TEST.WEIGHTS))
    # Create data loaders and meters
    test_loader = data_loader.construct_test_loader()
    test_meter = meters.TestMeter(len(test_loader))
    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)

    if cfg.ADV.VAL_ATTACK:
        test_adv_meter = meters.TestMeter(len(test_loader))
        test_adv_epoch(test_loader, model, test_adv_meter, 0)


def time_model():
    """Times model."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Compute model and loader timings
    benchmark.compute_time_model(model, loss_fun)


def time_model_and_loader():
    """Times model and data loader."""
    # Setup training/testing environment
    setup_env()
    # Construct the model and loss_fun
    model = setup_model()
    loss_fun = builders.build_loss_fun().cuda()
    # Create data loaders
    train_loader = data_loader.construct_train_loader()
    test_loader = data_loader.construct_test_loader()
    # Compute model and loader timings
    benchmark.compute_time_full(model, loss_fun, train_loader, test_loader)
