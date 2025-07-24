import torch
import torch.nn as nn
import numpy as np
import os, sys
from tqdm import tqdm
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.MEDIAR.utils import *
import matplotlib.pyplot as plt

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

    def create_center_mask(self, center_mask, radius=8):
        """
        Applies a circular dilation to center_mask (B, 1, H, W) using a disk of given radius.

        Args:
            center_mask (Tensor): [B, 1, H, W], binary map with 1s at detected cell centers
            radius (int): dilation radius (in pixels)

        Returns:
            Tensor: [B, 1, H, W] dilated mask
        """
        center_mask = center_mask[:,0]
        if len(center_mask.shape) == 3:
            center_mask = center_mask.unsqueeze(1) 
        B, _, H, W = center_mask.shape

        # Create circular disk kernel
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=center_mask.device),
            torch.arange(-radius, radius + 1, device=center_mask.device),
            indexing='ij'
        )
        disk = ((x**2 + y**2) <= radius**2).float()  # [2r+1, 2r+1]
        kernel = disk.view(1, 1, 2 * radius + 1, 2 * radius + 1)  # [1, 1, K, K]

        # Apply convolution to each image in the batch
        dilated = nn.functional.conv2d(center_mask.float(), kernel, padding=radius)
        dilated = (dilated > 0).float()  # binarize

        return dilated

    def mediar_criterion(self, outputs, labels_onehot_flows, center_mask=None, radius=8):
        """
        outputs: [B, C=3, H, W]
        labels_onehot_flows: numpy array of shape [B, C=3, H, W]
        center_mask: [B, 1, H, W] binary tensor of detected centers (from detection model), optional
        radius: radius around each center to supervise cellprob
        """

        # Move numpy inputs to tensors
        cellprob_target = torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float()  # [B, H, W]
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)  # [B, 2, H, W]

        # Cell Probability Prediction
        cellprob_pred = outputs[:, -1]  # [B, H, W]

        # ---- Cell Probability Loss (Masked if center_mask provided) ----
        if center_mask is not None:
            # [B, 1, H, W] -> [B, H, W]
            mask = self.create_center_mask(center_mask, radius=radius).squeeze(1)

            per_pixel_loss = nn.functional.binary_cross_entropy_with_logits(
                cellprob_pred, cellprob_target, reduction='none'
            )

            # Apply mask
            per_pixel_loss = per_pixel_loss.cpu()
            mask = mask.cpu()
            cellprob_loss = (per_pixel_loss * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            cellprob_loss = self.bce_loss(cellprob_pred, cellprob_target)

        # ---- Flow Loss (Masked to foreground regions only) ----
        flow_pred = outputs[:, :2]  # [B, 2, H, W]
        flow_mask = (cellprob_target > 0.5).float().unsqueeze(1).repeat(1, 2, 1, 1)  # [B, 2, H, W]

        flow_pred_masked = flow_pred * flow_mask
        gradient_flows_masked = 5.0 * gradient_flows * flow_mask

        # MSE loss only on foreground
        mse = nn.functional.mse_loss(flow_pred_masked, gradient_flows_masked, reduction='sum')
        denom = flow_mask.sum().clamp(min=1.0)
        gradflow_loss = 0.5 * (mse / denom)

        return cellprob_loss + gradflow_loss

    def _epoch_phase(self, phase):
        phase_results = {}

        self.loss_metric = []
        self.f1_metric = []
        self.iou_metric = []


        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)

            center_masks = batch_data.get("cellcenter", None)
            if center_masks is not None:
                # If it's a list of tensors, stack them to a batch tensor
                if isinstance(center_masks, list):
                    center_masks = torch.stack(center_masks, dim=0)
                center_masks = center_masks.to(self.device)

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
                        loss = self.mediar_criterion(outputs, labels_onehot_flows, center_masks)
                        self.loss_metric.append(loss)

                        # Calculate valid statistics
                        if phase != "train":
                            outputs, labels = self._post_process(outputs, labels)
                            iou_score, f1_score = self._get_metrics(outputs, labels)
                            self.f1_metric.append(f1_score)
                            self.iou_metric.append(iou_score)

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

        # After the loop, calculate average loss
        avg_loss = torch.stack(self.loss_metric).mean().item() if len(self.loss_metric) > 0 else None

        # Update phase_results with avg loss explicitly
        if len(self.loss_metric) > 0:
            avg_loss = torch.stack(self.loss_metric).mean().item()
        else:
            avg_loss = None
        phase_results["dice_loss"] = avg_loss

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_metric, "dice_loss", phase
        )
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )
            phase_results = self._update_results(
                phase_results, self.iou_metric, "iou", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""

        # images_np = images.detach().cpu().numpy()  # (B, C, H, W)
        # for i in range(images_np.shape[0]):
        #     img = images_np[i]
        #     if img.ndim == 3:
        #         # If shape is (C, H, W), select first channel for grayscale
        #         img = img[0]

        #     plt.figure(figsize=(4, 4))
        #     plt.imshow(img, cmap='gray')
        #     plt.title(f"Input Image {i}")
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

        # Run model inference
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
