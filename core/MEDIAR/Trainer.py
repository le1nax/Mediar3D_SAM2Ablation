import torch
import torch.nn as nn
import numpy as np
import os, sys
from tqdm import tqdm
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.MEDIAR.utils import *

__all__ = ["Trainer"]


class Trainer(BaseTrainer):
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
        super(Trainer, self).__init__(
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion,
            num_epochs,
            device,
            no_valid,
            valid_frequency,
            amp,
            algo_params,
        )

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def mediar_criterion(self, outputs, labels_onehot_flows):
        """loss function between true labels and prediction outputs"""

        # Cell Recognition Loss
        cellprob_loss = self.bce_loss(
            outputs[:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float(),
        )

        # Cell Distinction Loss
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)
        gradflow_loss = 0.5 * self.mse_loss(outputs[:, :2], 5.0 * gradient_flows)

        loss = cellprob_loss + gradflow_loss

        return loss

    def _epoch_phase(self, phase):
        phase_results = {}

        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images, labels = batch_data["img"], batch_data["label"]

            if self.with_public:
                # Load batches sequentially from the unlabeled dataloader
                try:
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                except:
                    # Assign memory loader if the cycle ends
                    self.public_iterator = iter(self.public_loader)
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                # Concat memory data to the batch
                images = torch.cat([images, images_pub], dim=0)
                labels = torch.cat([labels, labels_pub], dim=0)

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.amp):
                with torch.set_grad_enabled(phase == "train"):
                    # Output shape is B x [grad y, grad x, cellprob] x H x W
                    outputs = self._inference(images, phase)

                    # Map label masks to graidnet and onehot
                    labels_onehot_flows = labels_to_flows(
                        labels, use_gpu=True, device=self.device
                    )
                    # Calculate loss
                    loss = self.mediar_criterion(outputs, labels_onehot_flows)
                    self.loss_metric.append(loss)

                    # Calculate valid statistics
                    if phase != "train":
                        outputs, labels = self._post_process(outputs, labels)
                        f1_score = self._get_f1_metric(outputs, labels)
                        self.f1_metric.append(f1_score)

                # Backward pass
                if phase == "train":
                    # For the mixed precision training
                    if self.amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    else:
                        loss.backward()
                        self.optimizer.step()

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "dice_loss", phase
        )
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""

        if phase != "train":
            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            outputs = self.model(images)

        return outputs
    
    def _inference3D(self, img_data):
        """Conduct model prediction"""

        img_data = img_data.to(self.device)
        img_base = img_data
        #Lz, Ly, Lx = shape[:-1]
        ## @todo Anisotropy 
        # if anisotropy is not None and anisotropy != 1.0:
        #     models_logger.info(f"resizing 3D image with anisotropy={anisotropy}")
        #     x = transforms.resize_image(x.transpose(1,0,2,3),
        #                             Ly=int(Lz*anisotropy), 
        #                             Lx=int(Lx)).transpose(1,0,2,3)
        outputs_base = self.run_3D(img_base)
        cellprob = outputs_base[-1]
        dP = outputs_base[:-1]

        pred_mask = torch.cat([dP, cellprob.unsqueeze(0)], dim=0)

        pred_mask = pred_mask.squeeze() ##@todo cpu as in prediction?

        return pred_mask
    
    def run_3D(self, imgs): ###@todo channel adapt, batch size adapt
        
        #permute images  3012 3102 3201 (put 3 in first becazse window_inference wants NCHW)
        sstr = ["YX", "ZY", "ZX"]
        pm = [(3, 0, 1, 2), (3, 1, 0, 2), (3, 2, 0, 1)] 
        ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
        cp = [(1, 2), (0, 2), (0, 1)]
        cpy = [(0, 1), (0, 1), (0, 1)]
        shape = imgs.shape[:-1]
        yf = torch.zeros((4, *shape), dtype=torch.float32, device=self.device)
        for p in range(3):
            xsl = imgs.permute(pm[p]) ##images has now CZHW order
            # per image
            print("running %s: %d planes of size (%d, %d)" %
                            (sstr[p], shape[pm[p][1]], shape[pm[p][2]], shape[pm[p][3]]))
            
            outputs = []
            for z in range(shape[pm[p][1]]):  # iterate over Z
                slice_img = xsl[:, z, :, :].unsqueeze(0)  # shape (1, C, H, W) 
                out = self._window_inference(slice_img) #shape (3, HW)
                outputs.append(out.squeeze()) #remove 1st batch dim

            # Stack outputs along Z
            y = torch.stack(outputs, dim=1)  #shape(3, Z, H, W)

            y_p = y[-1].permute(ipm[p])
            yf[-1] += y_p
            for j in range(2):
                yf[cp[p][j]] += y[cpy[p][j]].permute(ipm[p])
            y = None; del y
    
        return yf

    def _post_process(self, outputs, labels=None):
        """Predict cell instances using the gradient tracking"""
        outputs = outputs.squeeze(0).cpu().numpy()
        gradflows, cellprob = outputs[:2], self._sigmoid(outputs[-1])
        outputs = compute_masks(gradflows, cellprob, use_gpu=True, device=self.device)
        outputs = outputs[0]  # (1, C, H, W) -> (C, H, W)

        if labels is not None:
            labels = labels.squeeze(0).squeeze(0).cpu().numpy()

        return outputs, labels

    def _sigmoid(self, z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))
