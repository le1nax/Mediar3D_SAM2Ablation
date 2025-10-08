import torch
import wandb
import pprint
import torch.distributed as dist

__all__ = ["print_learning_device", "print_with_logging"]


def print_learning_device(device):
    try:
        device_idx = int(device.split(":")[1])
    except Exception:
        device_idx = "N/A"
    log_device(f"Training on device {device} (GPU index {device_idx})")


def print_with_logging(results, step):
    """Print and log on the W&B server.

    Args:
        results (dict): results dictionary
        step (int): epoch index
    """
    if not dist.is_initialized() or dist.get_rank() == 0:
        # Print the results dictionary
        pp = pprint.PrettyPrinter(compact=True)
        pp.pprint(results)
        print()

        # Log on the w&b server
        wandb.log(results, step=step)

def log_device(message):
    """Log a message only from the main process in a distributed setting."""
    if not dist.is_initialized() or dist.get_rank() == 0:
            print(message)
