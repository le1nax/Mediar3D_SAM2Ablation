

import torch
import os
import wandb
import argparse, pprint
from datetime import datetime
import torch.distributed as dist

import os
os.environ["WANDB_MODE"] = "disabled"

def log_device(*args, **kwargs):
    """Drop-in replacement for print, only prints from rank 0 (or non-distributed)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

from train_tools import *
from SetupDict import TRAINER, OPTIMIZER, SCHEDULER, MODELS, PREDICTOR

# Ignore warnings for tiffle image reading
import logging

logging.getLogger().setLevel(logging.ERROR)

# Set torch base print precision
torch.set_printoptions(6)

def _get_setups(args, device, distributed=False, rank=0, world_size=1):
    model_args = args.train_setups.model
    model = MODELS[model_args.name](**model_args.params).to(device)

    if model_args.pretrained.enabled:
        weights = torch.load(model_args.pretrained.weights, map_location="cpu")
        print("\nLoading pretrained model....")
        model.load_state_dict(weights, strict=model_args.pretrained.strict)

    return model


def main(args):
    """Execute experiment."""
    log_device(os.getcwd())

    # --- DDP setup ---
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        distributed = True
    else:
        local_rank = 0
        world_size = 1
        distributed = False

    log_device("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))


    # Initialize W&B
    wandb.init(config=args, **args.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Fix randomness for reproducibility
    random_seeder(args.train_setups.seed)

    # Set experiment
    model = _get_setups(
        args,
        device=f"cuda:{local_rank}",
        distributed=distributed,
        rank=local_rank,
        world_size=world_size,
    )

    # Watch parameters & gradients of model
    wandb.watch(model, log="all", log_graph=True)
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
#parser.add_argument("--config_path", default="./config/step1_pretraining/phase1.json", type=str)
parser.add_argument("--config_path", default="./config/step2_finetuning/finetuning1.json", type=str)

args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Print configuration dictionary pretty
    pprint_config(opt)

    # Run experiment
    main(opt)
