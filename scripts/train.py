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
from typing import Union

import torch
import wandb
import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel

import meru.utils.distributed as dist
from meru.config import LazyConfig, LazyFactory
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager
from meru.utils.timer import Timer
from meru.utils.io import get_run_name
from meru.models import MERU, CLIPBaseline
from configs.test.linprobe_classification import evaluator as linprobe_clf_evaluator
from configs.test.zero_shot_classification import evaluator as zeroshot_clf_evaluator
from configs.test.zero_shot_retrieval import evaluator as zeroshot_retrieval_evaluator
from meru.utils.compare import compare_models


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
parser.add_argument(
    "--save-model", action="store_true",
    help="Whether to save the model after training."
)
# fmt: on


def initialize_wandb(
    model: Union[MERU, CLIPBaseline, DistributedDataParallel],
    _A: argparse.Namespace,
    _C: LazyConfig,
    ):
    """
    Initialize wandb run.
    """
    run_name = get_run_name(model)
    if _A.proj_layer_only:
        _C.model.embed_dim = _A.proj_layer_only
    
    wandb.login()
    wandb.init(
        project='meru', name=run_name,
        config=OmegaConf.to_container(_C, resolve=True),
    )
    

def evaluate_model(
    model: Union[MERU, CLIPBaseline, DistributedDataParallel],
    evaluators: dict[str, callable],
    ) -> dict[str, float]:
    
    all_eval_results = {}
    
    # Loop over all evaluators.
    for eval_name, evaluator in evaluators.items():
        
        # Run evalator.
        start_time = time.perf_counter()
        eval_results = evaluator(model)
        end_time = time.perf_counter()
        
        # Log results to terminal.
        header = ",".join(eval_results.keys())
        numbers = ",".join([f"{num:.1f}" for num in eval_results.values()])
        logger.info(f"\n{header}\n{numbers}")
        
        # Collect results in all_eval_results for logging to wandb.
        for score_name, score in eval_results.items():
            all_eval_results.update({f'{eval_name}/{score_name}': score})
        all_eval_results.update({f'{eval_name}/time': end_time - start_time})
    return all_eval_results


def get_log_str(
    output_dict: dict,
    train_timer: Timer,
    data_time
    ):
    timer_stats = (
        f"Iter {train_timer.iteration} | Time (sec): {data_time:.3f} data, "
        f"{train_timer.deltas[-1]:.3f} model | ETA: {train_timer.eta_hhmm}"
    )
    log_str = f"{timer_stats} [GPU {dist.gpu_mem_usage()} MB]"
    log_str += "\n\t\t\t| "
    log_str += f"[total_loss {output_dict['loss'].item():.3f}]"
    for key, value in output_dict["logging"].items():
        log_str += f" [{key} {value:.3f}]"
    return log_str

def get_train_results(output_dict, scheduler, scaler):
        train_results = {
            'train/loss': output_dict["loss"].item(),
            'lr': scheduler.get_last_lr()[0],
            'amp_scale': scaler.get_scale(),
            }
        for name, _loss in output_dict["logging"].items():
            train_results.update({f"train/{name}": _loss})
        return train_results
        

    
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
    
    # Copy original model for evaluation.
    import copy
    original_model = copy.deepcopy(model)
    
    evaluators = {
        "zero_shot_retrieval": instantiate(zeroshot_retrieval_evaluator),
        "linear_probe_classification": instantiate(linprobe_clf_evaluator),
        "zero_shot_classification": instantiate(zeroshot_clf_evaluator),
    }

    # Create an iterator from dataloader to sample batches perpetually.
    dataloader_iter = iter(dataloader)
    train_timer = Timer(
        start_iteration + 1, total_iterations=_C.train.num_iterations
        )
    
    # Use internal `module` for DDP.
    _model = model.module if isinstance(
        model, DistributedDataParallel
        ) else model
        
    # Freeze all layers except projection layer. Projection layer is
    # initialized with dimension specified by _A.proj_layer_only.
    if _A.proj_layer_only:
            
        # Freeze all params except for learnable params.
        learnable_params = set([
            'curv', 'visual_alpha', 'textual_alpha', 'logit_scale',
            'textual_proj.weight', 'visual_proj.weight',
        ])
        for name, param in _model.named_parameters():
            param.requires_grad = name in learnable_params

        # Initialize new projection layers.
        new_embd_dim = _A.proj_layer_only
        visual_out_dim = _model.visual_proj.weight.shape[1]
        textual_out_dim = _model.textual_proj.weight.shape[1]
               
        _model.visual_proj = torch.nn.Linear(
            visual_out_dim, new_embd_dim, bias=False, device=model.device
            )
        _model.textual_proj = torch.nn.Linear(
            textual_out_dim, new_embd_dim, bias=False, device=model.device
            )
        model.module = _model # is this needed?
        optimizer = LazyFactory.build_optimizer(_C, _model)
        
    assert compare_models(original_model, model), "Model has changed."
        
    # Create wandb run, only in main process.
    if dist.is_main_process():
        initialize_wandb(model, _A, _C)
        
    # Evaluate model before training.
    logger.info(f'Evaluating the model...')
    all_eval_results = evaluate_model(model, evaluators)
    wandb.log(all_eval_results, step=start_iteration)
    
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
        if iteration == start_iteration + 1 or iteration % _A.log_period == 0:
            log_str = get_log_str(output_dict, train_timer, data_time)
            logger.info(log_str)
            if dist.is_main_process():
                train_results = get_train_results(output_dict, scheduler, scaler)
                wandb.log(train_results, step=iteration)
            
        # Evaluate the model (only in main process).
        if dist.is_main_process() and iteration % _A.eval_period == 0:
            # Evaluate the final model outside train loop.
            if iteration != _C.train.num_iterations:
                logger.info(f'Evaluating the model...')
                all_eval_results = evaluate_model(model, evaluators)
                wandb.log(all_eval_results, step=iteration)

        # Save checkpoint to disk.
        if iteration % _A.checkpoint_period == 0 and dist.is_main_process():
            checkpoint_manager.step(iteration)

    # Save the final checkpoint.
    if dist.is_main_process():
        checkpoint_manager.final_step()
        
        if _A.save_model:
            run_name = run_name.replace('-', '_')
            path = Path('models') / f'{run_name}.pth'
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(model, DistributedDataParallel):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save(model_state_dict, path)
            
    # Evaluate the final model.
    if dist.is_main_process():
        logger.info(f'Evaluating the final model...')
        all_eval_results = evaluate_model(model, evaluators)
        wandb.log(all_eval_results, step=iteration)

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
