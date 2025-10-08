import torch
import os
import wandb
import argparse, pprint
from datetime import datetime
import torch.distributed as dist

import os
#os.environ["WANDB_MODE"] = "disabled"

def log_device(*args, **kwargs):
    """Drop-in replacement for print, only prints from rank 0 (or non-distributed)."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

from train_tools import *
from SetupDict import TRAINER, OPTIMIZER, SCHEDULER, MODELS, PREDICTOR
from Dataloading.datasetter import get_dataloaders_labeled_sampled

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
    if args.data_setups.labeled.sampling_ratios:
        dataloaders = get_dataloaders_labeled_sampled(
            **args.data_setups.labeled,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        )
    else:
        dataloaders = get_dataloaders_labeled(
            **args.data_setups.labeled,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )

    optimizer_args = args.train_setups.optimizer
    optimizer = OPTIMIZER[optimizer_args.name](
        model.parameters(), **optimizer_args.params
    )

    scheduler = None
    if args.train_setups.scheduler.enabled:
        scheduler_args = args.train_setups.scheduler
        scheduler = SCHEDULER[scheduler_args.name](optimizer, **scheduler_args.params)

    trainer_args = args.train_setups.trainer
    trainer = TRAINER[trainer_args.name](
        model,
        dataloaders,
        optimizer,
        scheduler,
        args.data_setups.labeled.incomplete_annotations,
        **trainer_args.params,
    )

    if args.data_setups.labeled.valid_portion == 0:
        trainer.no_valid = True


    return trainer

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
    trainer = _get_setups(
        args,
        device=f"cuda:{local_rank}",
        distributed=distributed,
        rank=local_rank,
        world_size=world_size,
    )
    if torch.cuda.device_count() > 1:
        trainer.model = torch.nn.parallel.DistributedDataParallel(
            trainer.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # Watch parameters & gradients of model
    wandb.watch(trainer.model, log="all", log_graph=True)

    # Conduct experiment
    # trainer.train()
    trainer.train()

    save_dir = "../../W_B/Hiera_PT/"
    os.makedirs(save_dir, exist_ok=True)  # make sure it exists

    # Current time string: e.g. '2025-07-11_18-25-42'
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save path
    model_path = os.path.join(save_dir, f"PT_condor_newdist_{current_time}.pth")
    log_device(f"Saving model to: {model_path}")
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            state_dict = (
            trainer.model.module.state_dict()  # unwrap if DDP
            if isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel)
            else trainer.model.state_dict()
            )
            torch.save(state_dict, model_path)
            log_device(f"Model successfully saved to {model_path}")

        except FileNotFoundError as e:
            log_device(f"FileNotFoundError: {e}")

        except PermissionError as e:
            log_device(f"PermissionError: {e}")

        except Exception as e:
            log_device(f"Unexpected error while saving model: {e}")

    # # Conduct prediction using the trained model
    # predictor = PREDICTOR[args.train_setups.trainer.name](
    #     trainer.model,
    #     args.train_setups.trainer.params.device,
    #     args.pred_setups.input_path,
    #     args.pred_setups.output_path,
    #     args.pred_setups.make_submission,
    #     args.pred_setups.exp_name,
    #     args.pred_setups.algo_params,
    # )

    # total_time = predictor.conduct_prediction()
    # wandb.log({"total_time": total_time})


# Parser arguments for terminal execution
parser = argparse.ArgumentParser(description="Config file processing")
parser.add_argument("--config_path", default="./config/step1_pretraining/phase1.json", type=str)
#parser.add_argument("--config_path", default="./config/step2_finetuning/finetuning1.json", type=str)
args = parser.parse_args()

#######################################################################################

if __name__ == "__main__":
    # Load configuration from .json file
    opt = ConfLoader(args.config_path).opt

    # Print configuration dictionary pretty
    pprint_config(opt)

    # Run experiment
    main(opt)
