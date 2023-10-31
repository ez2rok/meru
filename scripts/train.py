# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Train a MERU or CLIP model based on parameters specified by a config file.
"""
import argparse
import time
import random
from pathlib import Path

import torch
import wandb
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.cuda import amp

import meru.utils.distributed as dist
from meru.config import LazyConfig, LazyFactory
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager
from meru.utils.timer import Timer
from configs.test.linprobe_classification import evaluator as linprobe_clf_evaluator
from configs.test.zero_shot_classification import evaluator as zeroshot_clf_evaluator
from configs.test.zero_shot_retrieval import evaluator as zeroshot_retrieval_evaluator


# fmt: off
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--config", help="Path to a .py config file.")
parser.add_argument(
    "--output-dir", default="./output",
    help="Path to a directory to save checkpoints and job logs.",
)
parser.add_argument(
    "--resume", action="store_true",
    help="Whether to resume training from `--output-dir`. This script will find "
    "the last saved checkpoint and resume training. It is user's responsibility "
    "to provide matching config file in `--config`.",
)
parser.add_argument(
    '--proj-layer-only', type=int,
    help="Train a new projection layer with dimension specified by this "
    "argument. Freeze all parameters except for this new projection "
    "layer and learnable scalars (softmax temperature, curvature, text alpha "
    "scaling, and visual alpha scaling) for training stability."
)
parser.add_argument(
    "--checkpoint-period", type=int, default=5000, help="Checkpoint saving period."
)
parser.add_argument(
    "--log-period", type=int, default=100,
    help="Log to stdout/wandb periodically (only main process).",
)
parser.add_argument(
    '--eval-period', type=int, default=1000,
    help="Evaluate on validation set periodically (only main process).",
)
parser.add_argument(
    "--num-machines", type=int, default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus", type=int, default=0, help="Number of GPUs per machine."
)
parser.add_argument(
    "--machine-rank", type=int, default=0,
    help="Integer in [0, num_machines) to specifying machine ID.",
)
_random_port = random.randint(2000, 19999)
parser.add_argument(
    "--dist-url", default=f"tcp://127.0.0.1:{_random_port}",
    help="URL of the main process in distributed training, it defaults to "
    "localhost for single-machine training.",
)
parser.add_argument(
    "overrides", nargs="...", default=[], help="Config overrides (key-value pairs)."
)
# fmt: on


def main(_A: argparse.Namespace):
    # -------------------------------------------------------------------------
    #   BASIC SETUP FOR TRAINING JOB.
    # -------------------------------------------------------------------------
    # Create a config object and perform common setup.
    _C = LazyConfig.load(_A.config)
    _C = LazyConfig.apply_overrides(_C, _A.overrides)

    # Get process rank and world size (assuming distributed is initialized).
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()

    if getattr(_C.train, "seed", None) is None:
        _C.train.seed = int(time.time())

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(_C.train.seed + RANK)
    np.random.seed(_C.train.seed + RANK)
    torch.manual_seed(_C.train.seed + RANK)
    torch.backends.cudnn.deterministic = _C.train.cudnn_deterministic
    torch.backends.cudnn.benchmark = _C.train.cudnn_benchmark

    # Create output directory and save config in it.
    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LazyConfig.save(_C, output_dir / "config.yaml")

    # Create a logger for each process which writes to a separate log-file.
    logger.add(output_dir / f"log-rank{RANK}.txt", format="{time} {level} {message}")

    # Print process info, config and args.
    logger.info(f"Rank of current process: {RANK}. World size: {WORLD_SIZE}")
    logger.info(f"RANK {RANK} using random seed: {_C.train.seed + RANK}")
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    # -------------------------------------------------------------------------
    #   INSTANTIATE ALL OBJECTS FOR TRAINING.
    # -------------------------------------------------------------------------
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if _A.num_gpus != 0
        else torch.device("cpu")
    )
    dataloader = LazyFactory.build_dataloader(_C)
    model = LazyFactory.build_model(_C, device)
    optimizer = LazyFactory.build_optimizer(_C, model)
    scheduler = LazyFactory.build_lr_scheduler(_C, optimizer)
    scaler = amp.GradScaler(enabled=_C.train.amp)
    tokenizer = Tokenizer()
    
    checkpoint_manager = CheckpointManager(
        _A.output_dir,
        model=model,
        # optimizer=optimizer, # https://github.com/pytorch/pytorch/issues/40769#issuecomment-651854015
        scheduler=scheduler,
        scaler=scaler,
    )
    start_iteration = checkpoint_manager.resume() if _A.resume else 0
    
    evaluators = {
        # "zero_shot_classification": instantiate(zeroshot_clf_evaluator),
        "zero_shot_retrieval": instantiate(zeroshot_retrieval_evaluator),
        # "linear_probe_classification": instantiate(linprobe_clf_evaluator),
    }

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(dataloader)
    train_timer = Timer(start_iteration + 1, total_iterations=_C.train.num_iterations)
    
    _model = model.module if isinstance(
        model, torch.nn.parallel.distributed.DistributedDataParallel
        ) else model
        
    # Freeze all layers except projection layer. Projection layer is initialized
    # with dimension specified by _A.proj_layer_only.
    if _A.proj_layer_only:
            
        # Freeze all params except for learnable params.
        learnable_params = set([
            'curv', 'visual_alpha', 'textual_alpha', 'logit_scale',
            'textual_proj.weight', 'visual_proj.weight',
        ])
        for name, param in _model.named_parameters():
            param.requires_grad = name in learnable_params

        # Initialize new projection layers.
        proj_dim = _A.proj_layer_only
        visual_out_dim = _model.visual_proj.weight.shape[1]
        textual_out_dim = _model.textual_proj.weight.shape[1]
               
        _model.visual_proj = torch.nn.Linear(
            visual_out_dim, proj_dim, bias=False, device=model.device
            )
        _model.textual_proj = torch.nn.Linear(
            textual_out_dim, proj_dim, bias=False, device=model.device
            )
        optimizer = LazyFactory.build_optimizer(_C, model)
        
    # Create wandb run, only in main process.
    if dist.is_main_process():
        model_name = _model.__class__.__name__.lower()
        model_size = _C.model.visual.arch.split('_')[1]
        proj_dim = str(_model.visual_proj.weight.shape[0]).zfill(4)
        wandb.login()
        wandb.init(
            project='meru',
            name=f'{model_name}-vit-{model_size}-{proj_dim}',
            config=OmegaConf.to_container(_C, resolve=True),
        )
    
    # -------------------------------------------------------------------------
    #   TRAINING LOOP
    # -------------------------------------------------------------------------
    for iteration in range(start_iteration + 1, _C.train.num_iterations + 1):
        data_time = time.perf_counter()
        batch = next(dataloader_iter)
        data_time = time.perf_counter() - data_time

        model.train()
        train_timer.tic()
        optimizer.zero_grad()
        with amp.autocast(enabled=_C.train.amp):
            # Get image and text (tokens) from batch and pass through model.
            tokens = tokenizer(batch["text"])
            output_dict = model(batch["image"].to(device), tokens)
            loss = output_dict["loss"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_timer.toc()
        
        # Log training stats to terminal and wandb.
        if iteration == start_iteration or iteration % _A.log_period == 0:
            timer_stats = (
                f"Iter {train_timer.iteration} | Time (sec): {data_time:.3f} data, "
                f"{train_timer.deltas[-1]:.3f} model | ETA: {train_timer.eta_hhmm}"
            )
            log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
            log_str += "\n\t\t\t| "
            log_str += f"[total_loss {output_dict['loss'].item():.3f}]"
            for key, value in output_dict["logging"].items():
                log_str += f" [{key} {value:.3f}]"
            logger.info(log_str)
            
            if dist.is_main_process():
                wandb.log({'train/loss': output_dict["loss"].item()}, commit=False)
                for name, _loss in output_dict["logging"].items():
                    wandb.log({f"train/{name}": _loss}, commit=False)
                wandb.log({
                    'lr': scheduler.get_last_lr()[0],
                    'amp_scale': scaler.get_scale(),
                    },
                        step=iteration,
                    )
            
        # Log eval stats to terminal and wandb (only in main process).
        if dist.is_main_process() and (
            iteration == start_iteration or iteration % _A.eval_period == 0
            ):
            logger.info(f'Evaluating on validation set...')
            
            eval_stats = {}
            for eval_name, evaluator in evaluators.items():
                start_time = time.perf_counter()
                results_dict = evaluator(model)
                end_time = time.perf_counter()
                
                header = ",".join(results_dict.keys())
                numbers = ",".join([f"{num:.1f}" for num in results_dict.values()])
                logger.info(f"\n{header}\n{numbers}")
                
                for score_name, score in results_dict.items():
                    eval_stats.update({f'{eval_name}/{score_name}': score})
                eval_stats.update({f'{eval_name}/time': end_time - start_time})
            wandb.log(eval_stats, step=iteration)

        # Save checkpoint to disk.
        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

    # Save the final checkpoint.
    if dist.is_main_process():
        checkpoint_manager.final_step()

    # Close wandb run.
    wandb.finish()

if __name__ == "__main__":
    _A = parser.parse_args()
    if _A.num_gpus == 0:
        main(_A)
    else:
        # This will launch `main` and set appropriate CUDA device (GPU ID) as
        # per process (accessed in the beginning of `main`).
        dist.launch(
            main,
            num_machines=_A.num_machines,
            num_gpus_per_machine=_A.num_gpus,
            machine_rank=_A.machine_rank,
            dist_url=_A.dist_url,
            args=(_A,),
        )
