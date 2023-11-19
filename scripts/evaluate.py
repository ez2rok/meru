# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a trained model using implementations from `meru.evaluation` module.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from meru.config import LazyConfig, LazyFactory
from meru.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--config", help="Path to an evaluation config file (.py)")
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--save-eval-artifacts", action="store_true", help="If true, save evaluation artifacts.")
_AA("--save-eval-results", action="store_true", help="If true, save evaluation results.")
_AA("--proj", action="store_true", help="If true, apply projection layer.")
_AA("--norm", action="store_true",
    help="If true, apply normalization layer. In MERU, apply alpha scaling "
    "and exponential map. In CLIP, apply L2 normalization."
    )
_AA(
    "overrides", nargs="...", default=[],
    help="Config overrides (key-value pairs)."
    )

def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C.evaluator)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

    results_dict = evaluator(model, save=_A.save_eval_artifacts)
    if _A.save_eval_results:
        
        if isinstance(model, DistributedDataParallel):
            model = model.module
            
        # Convert all numpy floats to (standard Python) floats.
        results_dict = {
            k: v.item() if isinstance(v, np.float64) else v
            for k, v in results_dict.items()
            }
        
        # Make outpath.
        model_name = model.__class__.__name__.lower()
        model_size = 'small'
        embd_dim = str(model.textual_proj.weight.shape[0]).zfill(4)
        proj_str = 'P' if evaluator.proj else 'X'
        norm_str = 'N' if evaluator.norm else 'X'
        run_name = f'{model_name}_vit_{model_size}_{embd_dim}_E{proj_str}{norm_str}'
        outdir = Path("results") / run_name
        eval_name = evaluator.__str__().split('.')[-1].split(' ')[0].lower()
        outpath = outdir / f"{eval_name}_results.yaml"

        outpath.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            OmegaConf.create(results_dict),
            outpath,
        )

    # Log results for copy-pasting to spreadsheet, including checkpoint path.
    header = ",".join(results_dict.keys())
    numbers = ",".join([f"{num:.1f}" for num in results_dict.values()])

    logger.info(f"copypaste: {_A.checkpoint_path}")
    logger.info(f"\ncopypaste below:\n{header}\n{numbers}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
