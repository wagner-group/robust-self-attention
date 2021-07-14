#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Functions that handle saving and loading of checkpoints."""

import os

import pycls.core.distributed as dist
import torch
from pycls.core.config import cfg
from pycls.core.io import pathmgr
from pycls.core.net import unwrap_model
import pycls.core.logging as logging


# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"

# Checkpoints directory name
_DIR_NAME = "checkpoints"

logger = logging.get_logger(__name__)


def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_checkpoint_best():
    """Retrieves the path to the best checkpoint file."""
    return os.path.join(cfg.OUT_DIR, "model.pyth")


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not pathmgr.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in pathmgr.ls(checkpoint_dir))


def save_checkpoint(model, optimizer, epoch, best):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not dist.is_master_proc():
        return
    # Ensure that the checkpoint dir exists
    pathmgr.mkdirs(get_checkpoint_dir())
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    with pathmgr.open(checkpoint_file, "wb") as f:
        torch.save(checkpoint, f)
    # If best copy checkpoint to the best checkpoint
    if best:
        pathmgr.copy(checkpoint_file, get_checkpoint_best())
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert pathmgr.exists(checkpoint_file), err_str.format(checkpoint_file)
    with pathmgr.open(checkpoint_file, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    if "model_state" in checkpoint:
        ckpt = checkpoint["model_state"]
    elif "model" in checkpoint:
        ckpt = checkpoint["model"]
    else:
        ckpt = checkpoint

    # Skip loading fc layer if finetuning on different # classes
    fc_key = "head.fc"
    if cfg.MODEL.TYPE == "vision_transformer":
        fc_key = "head"

    weight_key = fc_key + ".weight"
    bias_key = fc_key + ".bias"
    if weight_key in ckpt and ckpt[weight_key].size(0) != cfg.MODEL.NUM_CLASSES:
        logger.info('Skipped loading fc layer due to mismatch between model and weights num classes.')
        del ckpt[weight_key]
        if bias_key in ckpt:
            del ckpt[bias_key]

    res = unwrap_model(model).load_state_dict(ckpt, strict=False)
    logger.info("Model load completed: {}".format(res))
    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"]) if optimizer else ()
    if "epoch" in checkpoint:
        return checkpoint["epoch"]
    return 0


def delete_checkpoints(checkpoint_dir=None, keep="all"):
    """Deletes unneeded checkpoints, keep can be "all", "last", or "none"."""
    assert keep in ["all", "last", "none"], "Invalid keep setting: {}".format(keep)
    checkpoint_dir = checkpoint_dir if checkpoint_dir else get_checkpoint_dir()
    if keep == "all" or not pathmgr.exists(checkpoint_dir):
        return 0
    checkpoints = [f for f in pathmgr.ls(checkpoint_dir) if _NAME_PREFIX in f]
    checkpoints = sorted(checkpoints)[:-1] if keep == "last" else checkpoints
    for checkpoint in checkpoints:
        pathmgr.rm(os.path.join(checkpoint_dir, checkpoint))
    return len(checkpoints)
