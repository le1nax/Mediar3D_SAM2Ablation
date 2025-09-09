import torch
import os
import wandb
import argparse, pprint
from datetime import datetime

import os
os.environ["WANDB_MODE"] = "disabled"


from train_tools import *
from SetupDict import TRAINER, OPTIMIZER, SCHEDULER, MODELS, PREDICTOR

# Ignore warnings for tiffle image reading
import logging

logging.getLogger().setLevel(logging.ERROR)

# Set torch base print precision
torch.set_printoptions(6)


def _get_setups(args):
    """Get experiment configuration"""

    # Set model
    model_args = args.train_setups.model
    model = MODELS[model_args.name](**model_args.params)

    # Load pretrained weights
    if model_args.pretrained.enabled:
        weights = torch.load(model_args.pretrained.weights, map_location="cpu")

        print("\nLoading pretrained model....")
        model.load_state_dict(weights, strict=model_args.pretrained.strict)

    # Set dataloaders
    dataloaders = get_dataloaders_labeled(**args.data_setups.labeled)

    # Set optimizer
    optimizer_args = args.train_setups.optimizer
    optimizer = OPTIMIZER[optimizer_args.name](
        model.parameters(), **optimizer_args.params
    )

    # Set scheduler
    scheduler = None

    if args.train_setups.scheduler.enabled:
        scheduler_args = args.train_setups.scheduler
        scheduler = SCHEDULER[scheduler_args.name](optimizer, **scheduler_args.params)

    # Set trainer
    trainer_args = args.train_setups.trainer
    trainer = TRAINER[trainer_args.name](
        model, dataloaders, optimizer, scheduler, args.data_setups.labeled.incomplete_annotations, **trainer_args.params
    )

    # Check if no validation
    if args.data_setups.labeled.valid_portion == 0:
        trainer.no_valid = True

    # Set public dataloader
    if args.data_setups.public.enabled:
        dataloaders = get_dataloaders_public(
            **args.data_setups.public.params
        )
        trainer.public_loader = dataloaders["public"]
        trainer.public_iterator = iter(dataloaders["public"])

    return trainer


def main(args):
    """Execute experiment."""
    print(os.getcwd())
    # Initialize W&B
    wandb.init(config=args, **args.wandb_setups)

    # How many batches to wait before logging training status
    wandb.config.log_interval = 10

    # Fix randomness for reproducibility
    random_seeder(args.train_setups.seed)

    # Set experiment
    trainer = _get_setups(args)

    # Watch parameters & gradients of model
    wandb.watch(trainer.model, log="all", log_graph=True)

    # Conduct experiment
    # trainer.train()
    trainer.train()

    save_dir = "../../W_B/Hiera_PT"
    os.makedirs(save_dir, exist_ok=True)  # make sure it exists

    # Current time string: e.g. '2025-07-11_18-25-42'
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save path
    model_path = os.path.join(save_dir, f"model_{current_time}.pth")
    print(f"Saving model to: {model_path}")
    try:
        os.makedirs(save_dir, exist_ok=True)  # ensure directory exists
        torch.save(trainer.model.state_dict(), model_path)
        print(f"Model successfully saved to {model_path}")

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")

    except PermissionError as e:
        print(f"PermissionError: {e}")

    except Exception as e:
        print(f"Unexpected error while saving model: {e}")

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
