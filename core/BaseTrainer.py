import torch
import numpy as np
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.metrics import CumulativeAverage
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

import os, sys
import copy
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.utils import print_learning_device, print_with_logging
from train_tools.measures import evaluate_metrics_cellseg


class BaseTrainer:
    """Abstract base class for trainer implementations"""

    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        scheduler=None,
        criterion=None,
        num_epochs=100,
        device="cuda:0",
        no_valid=False,
        valid_frequency=1,
        amp=False,
        algo_params=None,
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.no_valid = no_valid
        self.valid_frequency = valid_frequency
        self.device = device
        self.amp = amp
        self.best_weights = None
        self.best_f1_score = 0.1

        # FP-16 Scaler
        self.scaler = torch.cuda.amp.GradScaler() if amp else None

        # Assign algoritm-specific arguments
        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        # Cumulitive statistics
        self.loss_metric = CumulativeAverage()
        self.f1_metric = CumulativeAverage()
        self.iou_metric = CumulativeAverage()

        # Post-processing functions
        self.post_pred = Compose(
            [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
        )
        self.post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])

    def train_unfreezeme(self, freeze_epochs=30):
        """Train the model with optional encoder freezing"""

        print_learning_device(self.device)
        train_losses = []
        valid_losses = []

        # Freeze encoder
        print(f">>> Freezing encoder for first {freeze_epochs} epochs")
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # Only use trainable parameters for optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4
        )

        for epoch in range(1, self.num_epochs + 1):
            print(f"[Round {epoch}/{self.num_epochs}]")

            # Unfreeze encoder if needed
            if epoch == freeze_epochs + 1:
                print(f">>> Unfreezing encoder at epoch {epoch}")
                for param in self.model.encoder.parameters():
                    param.requires_grad = True

                # Re-initialize optimizer with all parameters
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
                
            if epoch % self.valid_frequency == 0 or epoch == 1:
                if not self.no_valid:
                    print(">>> Valid Epoch")
                    valid_results = self._epoch_phase("valid")
                    print_with_logging(valid_results, epoch)

                    valid_loss = valid_results.get("Valid_Dice_Loss", None)
                    if valid_loss is not None:
                        valid_losses.append(valid_loss)

                    if "Valid_F1_Score" in valid_results.keys():
                        current_f1_score = valid_results["Valid_F1_Score"]
                        self._update_best_model(current_f1_score)
                else:
                    print(">>> TuningSet Epoch")
                    tuning_cell_counts = self._tuningset_evaluation()
                    tuning_count_dict = {"TuningSet_Cell_Count": tuning_cell_counts}
                    print_with_logging(tuning_count_dict, epoch)

                    current_cell_count = tuning_cell_counts
                    self._update_best_model(current_cell_count)

            print("-" * 50)
            # Train Epoch Phase
            print(">>> Train Epoch")
            train_results = self._epoch_phase("train")
            print_with_logging(train_results, epoch)

            train_loss = train_results.get("Train_Dice_Loss", None)
            if train_loss is not None:
                train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()

        self.best_f1_score = 0

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train Loss")
        if len(valid_losses) > 0:
            valid_epochs = list(range(
                self.valid_frequency,
                self.valid_frequency * len(valid_losses) + 1,
                self.valid_frequency,
            ))
            plt.plot(valid_epochs, valid_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def train(self):
        """Train the model"""

        print_learning_device(self.device)
        train_losses = []
        valid_losses = []

        for epoch in range(1, self.num_epochs + 1):
            print(f"[Round {epoch}/{self.num_epochs}]")


            if epoch % self.valid_frequency == 0 and not epoch == 1:
                if not self.no_valid:
                    # Valid Epoch Phase
                    print(">>> Valid Epoch")
                    valid_results = self._epoch_phase("valid")
                    print_with_logging(valid_results, epoch)

                    valid_loss = valid_results.get("Valid_Dice_Loss", None)
                    if valid_loss is not None:
                        valid_losses.append(valid_loss)

                    if "Valid_F1_Score" in valid_results.keys():
                        current_f1_score = valid_results["Valid_F1_Score"]
                        self._update_best_model(current_f1_score)
                else:
                    print(">>> TuningSet Epoch")
                    tuning_cell_counts = self._tuningset_evaluation()
                    tuning_count_dict = {"TuningSet_Cell_Count": tuning_cell_counts}
                    print_with_logging(tuning_count_dict, epoch)

                    current_cell_count = tuning_cell_counts
                    self._update_best_model(current_cell_count)

            # Train Epoch Phase
            print(">>> Train Epoch")
            train_results = self._epoch_phase("train")
            print_with_logging(train_results, epoch)
            
            train_loss = train_results.get("Train_Dice_Loss", None)
            if train_loss is not None:
                train_losses.append(train_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            print("-" * 50)

            self.best_f1_score = 0

        # plt.figure(figsize=(8,5))
        # plt.plot(train_losses, label="Train Loss")
        # if len(valid_losses) > 0:
        #     # Note: valid losses may be recorded only every few epochs
        #     valid_epochs = list(range(self.valid_frequency, self.valid_frequency*len(valid_losses)+1, self.valid_frequency))
        #     plt.plot(valid_epochs, valid_losses, label="Validation Loss")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Training and Validation Loss")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            
    def _epoch_phase(self, phase): ################ OVERRIDDEN BY MEDIAR TRAINER
        """Learning process for 1 Epoch (for different phases).

        Args:
            phase (str): "train", "valid", "test"

        Returns:
            dict: statistics for the phase results
        """
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)
            self.optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.loss_metric.append(loss)

            # Backward pass
            if phase == "train":
                loss.backward()
                self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "loss", phase
        )

        return phase_results

    @torch.no_grad()
    def _tuningset_evaluation(self):
        cell_counts_total = []
        self.model.eval()

        for batch_data in tqdm(self.dataloaders["tuning"]):
            images = batch_data["img"].to(self.device)
            if images.shape[-1] > 5000:
                continue

            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
            )

            outputs = outputs.squeeze(0)
            outputs, _ = self._post_process(outputs, None)
            count = len(np.unique(outputs) - 1)
            cell_counts_total.append(count)

        cell_counts_total_sum = np.sum(cell_counts_total)
        print("Cell Counts Total: (%d)" % (cell_counts_total_sum))

        return cell_counts_total_sum

    def _update_results(self, phase_results, metric, metric_key, phase="train"):
        metric_key = "_".join([phase, metric_key]).title()

        if isinstance(metric, list):
            if len(metric) > 0:
                # Convert elements to tensors if possible
                tensor_metrics = []
                for m in metric:
                    if isinstance(m, np.ndarray):
                        tensor_metrics.append(torch.from_numpy(m))
                    elif isinstance(m, float):
                        tensor_metrics.append(torch.tensor(m))
                    elif isinstance(m, torch.Tensor):
                        tensor_metrics.append(m)
                    else:
                        # fallback, convert to float then tensor
                        tensor_metrics.append(torch.tensor(float(m)))

                metric_item = round(torch.stack(tensor_metrics).mean().item(), 4)
            else:
                metric_item = None
        else:
            metric_item = round(metric.aggregate().item(), 4)
            metric.reset()

        phase_results[metric_key] = metric_item

        return phase_results

    def _update_best_model(self, current_f1_score):
        if current_f1_score > self.best_f1_score:
            self.best_weights = copy.deepcopy(self.model.state_dict())
            self.best_f1_score = current_f1_score
            print(
                "\n>>>> Update Best Model with score: {}\n".format(self.best_f1_score)
            )
        else:
            pass

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""
        if phase != "train":
            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="reflect",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            outputs = self.model(images)

        return outputs

    def _post_process(self, outputs, labels):
        return outputs, labels

    def _get_metrics(self, masks_pred, masks_true):
        iou, p,r, f1 = evaluate_metrics_cellseg(masks_pred, masks_true)

        return iou, f1