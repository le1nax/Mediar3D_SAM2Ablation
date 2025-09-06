import torch
import torch.nn as nn

import torch.nn.functional as F
from scipy.ndimage import binary_dilation

import numpy as np
import os, sys
from tqdm import tqdm
from monai.inferers import sliding_window_inference

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BaseTrainer import BaseTrainer
from core.MEDIAR.utils import *
from monai.losses import DiceLoss

import tifffile as tiff

__all__ = ["Trainer"]

def pad_to_multiple(tensor, multiple=32):
    _, h, w = tensor.shape  # tensor shape is [C, H, W]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

def plot_image(image, title='Image', slice_idx=None, cmap='gray'):
    """
    Plot a 2D image or a slice from a 3D image.

    Parameters:
        image (np.ndarray): Image data (2D or 3D).
        title (str): Plot title.
        slice_idx (int): Index of the slice to show if image is 3D. If None, shows middle slice.
        cmap (str): Colormap to use.
    """
    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

    elif image.ndim == 3:
        if slice_idx is None:
            slice_idx = image.shape[0] // 2
        plt.imshow(image[slice_idx], cmap=cmap)
        plt.title(f"{title} (slice {slice_idx})")
        plt.axis('off')
        plt.show()

    else:
        raise ValueError("Image must be 2D or 3D numpy array.")
    

def plot_overlay_image(image1, image2, title='Overlay Image', slice_idx=None, alpha=0.5):
    """
    Plot a 2D image with a blue-transparent overlay from another image.
    
    Parameters:
        image1 (np.ndarray): Base image (2D or 3D), shown in 'magma'.
        image2 (np.ndarray): Overlay image (same shape), shown in blue with transparency.
        title (str): Plot title.
        slice_idx (int): Index of the slice to show if 3D. If None, uses middle slice.
        alpha (float): Opacity of the overlay image (0 to 1).
    """
    # Handle 3D inputs
    if image1.ndim == 3:
        if slice_idx is None:
            slice_idx = image1.shape[0] // 2
        image1 = image1[slice_idx]
        image2 = image2[slice_idx]

    if image1.shape != image2.shape:
        raise ValueError("image1 and image2 must have the same shape.")

    plt.figure(figsize=(6, 6))
    plt.imshow(image1, cmap='Blues')
    plt.imshow(image2, cmap='magma', alpha=alpha)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def show_QC_results(src_image, pred_image, gt_image):
    """
    Show quality control results for a single 2D slice.

    Parameters:
        src_image (ndarray): 2D source image (grayscale)
        pred_image (ndarray): 2D predicted segmentation mask (binary or probabilistic)
        gt_image (ndarray): 2D ground truth mask (binary or probabilistic)
        cellseg_metric (dict, optional): Dictionary of metric results to optionally show.
    """
    # Normalize input image
    norm = mcolors.Normalize(vmin=np.percentile(src_image, 1), vmax=np.percentile(src_image, 99))
    mask_norm = mcolors.Normalize(vmin=0, vmax=1)

    # Set up figure and axes
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))  # Adjusted size for readability

    # 1. Source image
    axes[0].imshow(src_image, norm=norm, cmap='magma', interpolation='nearest')
    axes[0].set_title('Source Image')

    # 2. Overlay: Source + Prediction
    axes[1].imshow(src_image, norm=norm, cmap='magma', interpolation='nearest')
    axes[1].imshow(pred_image, norm=mask_norm, alpha=0.5, cmap='Blues')
    axes[1].set_title('Overlay: Source + Prediction')

    # 3. Prediction only
    axes[2].imshow(pred_image, cmap='Blues', norm=mask_norm, interpolation='nearest')
    axes[2].set_title('Prediction')

    # 4. Ground Truth
    axes[3].imshow(gt_image, interpolation='nearest', norm=mask_norm, cmap='Greens')


    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def compare_flows(flow_pred, flow_loaded, atol=1e-5, rtol=1e-3):
    """
    Compares two flow tensors or arrays. Assumes both are [B, C, H, W].
    Automatically converts numpy arrays to tensors and moves them to the same device.
    """
    if isinstance(flow_pred, np.ndarray):
        flow_pred = torch.from_numpy(flow_pred)

    if isinstance(flow_loaded, np.ndarray):
        flow_loaded = torch.from_numpy(flow_loaded)

    # Ensure same device
    flow_pred = flow_pred.to(flow_loaded.device)

    if flow_pred.shape != flow_loaded.shape:
        print(f"Shape mismatch: predicted {flow_pred.shape}, loaded {flow_loaded.shape}")
        return False

    equal = torch.allclose(flow_pred, flow_loaded, atol=atol, rtol=rtol)

    if not equal:
        diff = (flow_pred - flow_loaded).abs()
        print(f"Flow mismatch! Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
    else:
        print("Flows match.")
    
    return equal

def plot_imageSlider(image, cmap='gray', title=''):
    

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)  # Leave space for slider

    # Display image with log scale
    im = ax.imshow(image, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1))
    fig.colorbar(im, ax=ax, label='Cell Probability')
    ax.set_title('Interactive Thresholding')
    ax.axis('off')

    # Add a slider axis below the plot
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
    threshold_slider = Slider(
        ax=ax_slider,
        label='Threshold',
        valmin=1e-4,
        valmax=1,
        valinit=1e-4,
        valstep=1e-4
    )

    # Update function for the slider
    def update(val):
        threshold = threshold_slider.val
        masked_image = np.copy(image)
        masked_image[masked_image < threshold] = 0
        im.set_data(masked_image)
        fig.canvas.draw_idle()

    # Connect slider to update function
    threshold_slider.on_changed(update)

    # Show the window
    plt.show()
        

def _sigmoid(z):
    return 1 / (1 + np.exp(-z))



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
        current_bsize=1,
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

        self.current_bsize = current_bsize
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.loss_history = {
            "epoch": [],
            "total_loss": [],
            "cellprob_bce": [],
            "flow_mse": [],
            "dice_loss": [],
            "total_loss": [],
            "val_loss": [],
            "val_iou": [],
        }

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
   

    # def mediar_criterion(self, outputs, labels_onehot_flows, dilation_iters=10):
    #     """Loss function between true labels and prediction outputs with partial annotations support."""

    #     # --- Ensure tensor ---
    #     if isinstance(labels_onehot_flows, np.ndarray):
    #         labels_onehot_flows = torch.from_numpy(labels_onehot_flows).to(self.device)
    #     else:
    #         labels_onehot_flows = labels_onehot_flows.to(self.device)

    #     # --- Build ground truth tensors ---
    #     gt_cellprob = (labels_onehot_flows[:, 1] > 0.5).float()   # (B,H,W)
    #     gt_flows = labels_onehot_flows[:, 2:].float()             # (B,2,H,W)

    #     # --- Supervision mask (initially: only where annotations exist) ---
    #     supervision_mask = gt_cellprob.clone()

    #     # --- Special case: background-only slices (no labels) ---
    #     if supervision_mask.sum() == 0:
    #         # Use full image as supervision mask
    #         supervision_mask = torch.ones_like(supervision_mask, device=self.device)

    #     # --- Dilate mask if needed ---
    #     elif dilation_iters > 0:
    #         mask_np = supervision_mask.cpu().numpy()
    #         mask_np = np.stack([binary_dilation(m, iterations=dilation_iters) for m in mask_np])
    #         supervision_mask = torch.from_numpy(mask_np).to(self.device).float()

    #     # --- Cell Recognition Loss (BCE masked) ---
    #     raw_bce = F.binary_cross_entropy_with_logits(outputs[:, -1], gt_cellprob, reduction="none")
    #     cellprob_loss = (raw_bce * supervision_mask).sum() / (supervision_mask.sum() + 1e-6)

    #     # --- Cell Distinction Loss (Flow masked MSE) ---
    #     raw_mse = F.mse_loss(outputs[:, :2], 5.0 * gt_flows, reduction="none")  # (B,2,H,W)
    #     mask_flows = supervision_mask.unsqueeze(1)  # (B,1,H,W)
    #     gradflow_loss = (raw_mse * mask_flows).sum() / (mask_flows.sum() + 1e-6)

    #     return cellprob_loss, 0.05 * gradflow_loss
    

    def mediar_criterion(self, outputs, labels_onehot_flows):
        """loss function between true labels and prediction outputs"""

        # Cell Recognition Loss
        cellprob_loss = self.bce_loss(
            outputs[:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float(),
        )

        # Cell Distinction Loss
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)
        gradflow_loss = self.mse_loss(outputs[:, :2], gradient_flows*5)

        if torch.isnan(cellprob_loss):
            asdf = 123  #debug
            #cellprob_loss = torch.tensor(0.0, device=self.device)

        return cellprob_loss, 0.5* gradflow_loss
    
    def _crop_to_ROI(self, images, labels, flows=None, center_masks=None):
        """
        Crop each image in the batch to the ROI of its label OR keep full image with probability full_prob.
        Always pads to nearest multiple of 32.
        
        images, labels, center_masks shape: [B, C, H, W]
        flows shape (if given): [B, H, W, C]
        """
        cropped_images, cropped_labels = [], []
        cropped_center_masks, cropped_flows = [], []

        for b in range(self.current_bsize):
            label = labels[b, 0]  # [H, W]
            nonzero = (label > 0).nonzero(as_tuple=False)

            # case 1: empty label -> keep full image
            if nonzero.shape[0] == 0:
                cropped_images.append(images[b])
                cropped_labels.append(labels[b])
                if center_masks is not None:
                    cropped_center_masks.append(center_masks[b])
                if flows is not None:
                    cropped_flows.append(flows[b])
                continue

            # # case 2: non-empty, maybe keep full image
            # if random.random() < full_prob:
            #     cropped_images.append(images[b])
            #     cropped_labels.append(labels[b])
            #     if center_masks is not None:
            #         cropped_center_masks.append(center_masks[b])
            #     if flows is not None:
            #         cropped_flows.append(flows[b])
            #     continue

            # case 3: ROI crop
            y_min, y_max = nonzero[:, 0].min().item(), nonzero[:, 0].max().item()
            x_min, x_max = nonzero[:, 1].min().item(), nonzero[:, 1].max().item()

            buffer = 20
            H, W = label.shape
            y_start = max(y_min - buffer, 0)
            y_end   = min(y_max + buffer, H)
            x_start = max(x_min - buffer, 0)
            x_end   = min(x_max + buffer, W)

            cropped_images.append(images[b, :, y_start:y_end, x_start:x_end])
            cropped_labels.append(labels[b, :, y_start:y_end, x_start:x_end])
            if center_masks is not None:
                cropped_center_masks.append(center_masks[b, :, y_start:y_end, x_start:x_end])
            if flows is not None:
                cropped_flows.append(flows[b, y_start:y_end, x_start:x_end, :])  # [H,W,C]

        # --- Compute max dims ---
        all_heights = [img.shape[1] for img in cropped_images]
        all_widths  = [img.shape[2] for img in cropped_images]

        if flows is not None:
            all_heights += [flow.shape[0] for flow in cropped_flows]  # [H,W,C]
            all_widths  += [flow.shape[1] for flow in cropped_flows]

        max_h, max_w = max(all_heights), max(all_widths)

        # Round up to nearest multiple of 32
        pad_h = ((max_h + 31) // 32) * 32
        pad_w = ((max_w + 31) // 32) * 32

        # --- Pad helper ---
        def pad_tensor(tensor, is_channels_last=False):
            if is_channels_last:
                # tensor shape: [H, W, C]
                h, w, c = tensor.shape
                pad_top = (pad_h - h) // 2
                pad_bottom = pad_h - h - pad_top
                pad_left = (pad_w - w) // 2
                pad_right = pad_w - w - pad_left
                padded = torch.nn.functional.pad(
                    tensor.permute(2, 0, 1),
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', value=0
                )
                return padded.permute(1, 2, 0)  # back to [H, W, C]
            else:
                # tensor shape: [C, H, W]
                c, h, w = tensor.shape
                pad_top = (pad_h - h) // 2
                pad_bottom = pad_h - h - pad_top
                pad_left = (pad_w - w) // 2
                pad_right = pad_w - w - pad_left
                return torch.nn.functional.pad(
                    tensor,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant', value=0
                )

        # --- Pad everything ---
        padded_images = [pad_tensor(img, is_channels_last=False) for img in cropped_images]
        padded_labels = [pad_tensor(lbl, is_channels_last=False) for lbl in cropped_labels]
        padded_center_masks = [pad_tensor(center, is_channels_last=False) for center in cropped_center_masks] if center_masks is not None else None
        padded_flows = [pad_tensor(flow, is_channels_last=True) for flow in cropped_flows] if flows is not None else None

        # --- Stack ---
        images = torch.stack(padded_images)
        labels = torch.stack(padded_labels)
        center_masks = torch.stack(padded_center_masks) if center_masks is not None else None
        flows = torch.stack(padded_flows) if flows is not None else None

        return images, labels, flows


    def _epoch_phase(self, phase):
        phase_results = {}

        self.loss_flow = []
        self.loss_cellprob = []
        self.f1_metric = []
        self.iou_metric = []
        self.loss_dice = []


        # Set model mode
        self.model.train() if phase == "train" else self.model.eval()

        qc_counter = 0  # Reset at the beginning of each phase
        
        # Epoch process
        for batch_data in tqdm(self.dataloaders[phase]):
            images = batch_data["img"].to(self.device)
            labels = batch_data["label"].to(self.device)
            self.current_bsize = images.shape[0]
            flows = batch_data.get("flow", None)
            if flows is not None:
                # If flows is a list of file paths (str), load them
                if isinstance(flows[0], str):       
                    flows = [torch.from_numpy(tiff.imread(f)).float().to(self.device) for f in flows]
                if isinstance(flows, list):
                    flows = torch.stack(flows, dim=0)


            center_masks = batch_data.get("cellcenter", None)
            if center_masks is not None:
                # If it's a list of tensors, stack them to a batch tensor
                if isinstance(center_masks, list):
                    center_masks = torch.stack(center_masks, dim=0)
                center_masks = center_masks.to(self.device)

            if self.with_public: #@todo add precomputed flows
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

            #plot_image(images[0].cpu().numpy())
            #plot_image(labels[0].cpu().numpy())
            #images, labels, flows = self._crop_to_ROI(images, labels, flows)
            #plot_image(images[0].cpu().numpy())
            #plot_image(labels[0].cpu().numpy())
            
            self.optimizer.zero_grad()
            # Forward pass
            with torch.amp.autocast(device_type="cuda", enabled=False):#self.amp):
                with torch.set_grad_enabled(phase == "train"):
                    # Output shape is B x [grad y, grad x, cellprob] x H x W
                    outputs = self._inference(images, phase)
                    #plot_image(outputs[-1].cpu().detach().numpy())

                    # Map label masks to graidnet and onehot
                    labels_onehot_flows = labels_to_flows(
                        labels, use_gpu=True, device=self.device
                    )

                    # compare_flows(labels_onehot_flows, flows.to(self.device))
                    #plot_image(_sigmoid(outputs[0,0,:,:].cpu().detach().numpy()))
                    # if qc_counter % 50 == 0:
                    #     show_QC_results(images[0,0].cpu().numpy(), _sigmoid(outputs[0,-1,:,:].cpu().detach().numpy()), labels[0,-1].cpu().numpy())
                        
                    
                    # Calculate loss
                    loss_prob, loss_flow = self.mediar_criterion(outputs, labels_onehot_flows)
                    loss = loss_prob + loss_flow
                    self.loss_flow.append(loss_flow)
                    self.loss_cellprob.append(loss_prob)

                    # Calculate valid statistics
                    if phase == "train" and qc_counter % 800 == 0:
                        outputs, labels = self._post_process(outputs.detach(), center_masks, labels)
                        for b in range(self.current_bsize):
                            iou_score, f1_score = self._get_metrics(outputs[b], labels[b])
                            print(f"  [Train QC]  F1: {f1_score:.3f}, IoU: {iou_score:.3f}")

                    # Calculate valid statistics
                    if phase != "train":
                        outputs, labels = self._post_process(outputs, center_masks, labels)

                        # plot_image(outputs)
                        # plot_image(labels)

                        for b in range(self.current_bsize):
                            iou_score, f1_score = self._get_metrics(outputs[b], labels[b])
                            self.f1_metric.append(f1_score)
                            self.iou_metric.append(iou_score)

                        # if qc_counter % 50 == 0:
                        #     show_QC_results(images[0, 0].cpu(), outputs[:,:], labels)

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
            qc_counter += 1

               

        # Update metrics
        phase_results = self._update_results(
            phase_results, self.loss_flow, "flow_loss", phase
        )
        phase_results = self._update_results(
            phase_results, self.loss_cellprob, "cellprob bce loss", phase
        )
        if phase != "train":
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )
            phase_results = self._update_results(
                phase_results, self.iou_metric, "iou", phase
            )

        # Track epoch number
        epoch_idx = len(self.loss_history["epoch"])
        self.loss_history["epoch"].append(epoch_idx)

        # Store metrics
        bce = torch.stack(self.loss_cellprob).mean().item()
        mse = torch.stack(self.loss_flow).mean().item()
        total_loss = bce + mse

        if phase == "train":
            self.loss_history["cellprob_bce"].append(bce)
            self.loss_history["flow_mse"].append(mse)
            self.loss_history["total_loss"].append(total_loss)

        # Store iou if available
        if phase != "train":
            self.loss_history["val_iou"].append(np.mean(self.iou_metric))
            self.loss_history["val_loss"].append(np.mean(total_loss))
        else:
            self.loss_history["val_iou"].append(np.nan)
            self.loss_history["val_loss"].append(np.nan)

        # # Plot if valid step
        # if phase != "train":
        #     self._plot_loss_metrics()


        
        return phase_results

    def _inference(self, images, phase="train"):
        """inference methods for different phase"""

        #images_np = images.detach().cpu().numpy()  # (B, C, H, W)
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


    def _plot_loss_metrics(self):
        import numpy as np
        import matplotlib.pyplot as plt

        history = self.loss_history

        # Extract lists
        epochs = np.array(history["epoch"])
        train_total = np.array(history["total_loss"], dtype=np.float32)
        val_total = np.array(history["val_loss"], dtype=np.float32)
        val_iou = np.array(history["val_iou"], dtype=np.float32)

        # Get minimum length to align all lists
        min_len = min(len(epochs), len(train_total), len(val_total), len(val_iou))

        # Truncate to same length
        epochs = epochs[:min_len]
        train_total = train_total[:min_len]
        val_total = val_total[:min_len]
        val_iou = val_iou[:min_len]

        # Remove NaNs
        mask = ~np.isnan(train_total) & ~np.isnan(val_total) & ~np.isnan(val_iou)
        epochs = epochs[mask]
        train_total = train_total[mask]
        val_total = val_total[mask]
        val_iou = val_iou[mask]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_total, label="Train Total Loss", color="orange", linewidth=2)
        plt.plot(epochs, val_total, label="Val Total Loss", color="blue", linewidth=2)
        plt.plot(epochs, val_iou, label="Val IOU", color="green", linestyle="--", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Train/Val Total Loss and IOU")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
            
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

    def _post_process(self, outputs, cellcenters=None,labels=None):
        """Predict cell instances using the gradient tracking"""
        outputs_batch = []
        outputs = outputs.cpu().numpy()  # (B, C, H, W)
        for b in range(self.current_bsize):
            outputs_b = outputs[b]
            gradflows, cellprob = outputs_b[:2], self._sigmoid(outputs_b[-1])
            outputs_b = compute_masks(gradflows, cellprob, use_gpu=True, device=self.device)
            outputs_b = outputs_b[0]  # (1, C, H, W) -> (C, H, W)
            outputs_batch.append(outputs_b)
            # if(cellcenters is not None):
            # outputs = filter_false_positives(outputs, cellcenters)
        outputs = np.stack(outputs_batch, axis=0)  # (B, C, H, W)

        if labels is not None:
            labels = labels.squeeze(1).cpu().numpy()

        return outputs, labels


    def _sigmoid(self, z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))
