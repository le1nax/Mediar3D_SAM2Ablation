import torch
import numpy as np
import os, sys
from monai.inferers import sliding_window_inference
from skimage import morphology, measure
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from core.BasePredictor import BasePredictor
from core.MEDIAR.utils import compute_masks, compute_masks3D, filter_false_positives

__all__ = ["Predictor"]

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


class Predictor(BasePredictor):
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        cellcenters_path=None,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        super(Predictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            cellcenters_path,
            make_submission,
            exp_name,
            algo_params,
        )
        self.hflip_tta = HorizontalFlip()
        self.vflip_tta = VerticalFlip()

    @torch.no_grad()
    def _inference(self, img_data):
        """Conduct model prediction"""

        img_data = img_data.to(self.device)
        img_base = img_data
        outputs_base = self._window_inference(img_base) 
        #outputs_base = self.model(img_base)

        outputs_base = outputs_base.cpu().squeeze()
        img_base.cpu()

        if not self.use_tta:
            pred_mask = outputs_base
            return pred_mask

        else:
            # HorizontalFlip TTA
            img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)
            outputs_hflip = self._window_inference(img_hflip)
            outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
            outputs_hflip = outputs_hflip.cpu().squeeze()
            img_hflip = img_hflip.cpu()

            # VertricalFlip TTA
            img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)
            outputs_vflip = self._window_inference(img_vflip)
            outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
            outputs_vflip = outputs_vflip.cpu().squeeze()
            img_vflip = img_vflip.cpu()

            # Merge Results
            pred_mask = torch.zeros_like(outputs_base)
            pred_mask[0] = (outputs_base[0] + outputs_hflip[0] - outputs_vflip[0]) / 3
            pred_mask[1] = (outputs_base[1] - outputs_hflip[1] + outputs_vflip[1]) / 3
            pred_mask[2] = (outputs_base[2] + outputs_hflip[2] + outputs_vflip[2]) / 3

        return pred_mask

    def _window_inference(self, img_data, aux=False):
        """Inference on RoI-sized window"""

        #img_data expecting NCHW shape
        y = sliding_window_inference(
            img_data,
            roi_size=128,
            sw_batch_size=4,
            predictor=self.model if not aux else self.model_aux,
            padding_mode="constant",
            mode="gaussian", ###@todo: oom
            overlap=0.6,
        )

        return y

    def _post_process(self, pred_mask, cellcenters):
        """Generate cell instance masks."""
        dP, cellprob = pred_mask[:2], self._sigmoid(pred_mask[-1])
        H, W = pred_mask.shape[-2], pred_mask.shape[-1]

        if np.prod(H * W) < (5000 * 5000):
            pred_mask = compute_masks(
                dP,
                cellprob,
                use_gpu=True,
                flow_threshold=0.4,
                device=self.device,
                cellprob_threshold=0.5,
            )[0]
        
        else:
            print("\n[Whole Slide] Grid Prediction starting...")
            roi_size = 2000

            # Get patch grid by roi_size
            if H % roi_size != 0:
                n_H = H // roi_size + 1
                new_H = roi_size * n_H
            else:
                n_H = H // roi_size
                new_H = H

            if W % roi_size != 0:
                n_W = W // roi_size + 1
                new_W = roi_size * n_W
            else:
                n_W = W // roi_size
                new_W = W

            # Allocate values on the grid
            pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
            dP_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
            cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

            dP_pad[:, :H, :W], cellprob_pad[:H, :W] = dP, cellprob

            for i in range(n_H):
                for j in range(n_W):
                    print("Pred on Grid (%d, %d) processing..." % (i, j))
                    dP_roi = dP_pad[
                        :,
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                    cellprob_roi = cellprob_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]

                    pred_mask = compute_masks(
                        dP_roi,
                        cellprob_roi,
                        use_gpu=True,
                        flow_threshold=0.4,
                        device=self.device,
                        cellprob_threshold=0.5,
                    )[0]

                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ] = pred_mask

            pred_mask = pred_pad[:H, :W]
        
        if(cellcenters is not None and cellcenters != ""):
            pred_mask = filter_false_positives(pred_mask, cellcenters)
        return pred_mask

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    


    def plot_image(self, image, cmap='gray', title=''):
        """
        Plots a 2D image using matplotlib.

        Parameters:
        - image: 2D numpy array representing the image.
        - cmap: Colormap to use (default is 'gray').
        - title: Optional title for the plot.
        """
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='magma', norm=LogNorm(vmin=1e-4, vmax=1))  # adjust vmin as needed
        plt.title(title)
        plt.colorbar(label='Cell Probability')
        plt.axis('off')
        plt.show()

    def plot_imageSlider(self, image, cmap='gray', title=''):
    

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
        


    def run_3D(self, imgs):
        """
        Memory-efficient 3D inference (slice-by-slice).

        Accepts imgs in either:
        - (C, Z, Y, X)  <-- preferred (channel-first)
        - (Z, Y, X, C)  <-- will be converted automatically

        Returns:
        yf: torch.Tensor shape (4, Z, Y, X) on CPU (float32).
        Channels: 3 flow maps + probability map (aggregated / averaged).
        """

        # --- normalize to (C, Z, Y, X) ---
        if imgs.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {imgs.shape}")
        if imgs.shape[0] in (1, 2, 3, 4):
            vol = imgs.contiguous()
        elif imgs.shape[-1] in (1, 2, 3, 4):
            vol = imgs.permute(3, 0, 1, 2).contiguous()
        else:
            # fallback assume channel-first
            vol = imgs.contiguous()

        C, Z, Y, X = vol.shape

        # accumulators on CPU (float32 for numeric safety)
        yf = torch.zeros((4, Z, Y, X), dtype=torch.float32, device="cpu")

        # permutations adapted for channel-first input (produce (C, num_planes, H, W))
        pm = [
            (0, 1, 2, 3),  # Z-slices -> (C, Z, Y, X)
            (0, 2, 1, 3),  # Y-slices -> (C, Y, Z, X)
            (0, 3, 1, 2),  # X-slices -> (C, X, Z, Y)
        ]

        ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]

        # which global flow channels to add local in-plane flows to (same as your original)
        cp = [(1, 2), (0, 2), (0, 1)]

        cpy = [(0, 1), (0, 1), (0, 1)]

        model_device = next(self.model.parameters()).device

        with torch.no_grad():
            for p in range(3):
                # if p != 1:
                #     continue
                xsl = vol.permute(pm[p]).contiguous()  # (C, num_planes, H, W)
                num_planes = xsl.shape[1]
                H = xsl.shape[2]; W = xsl.shape[3]
                
                print(f"[run_3D] plane {p}: num_planes={num_planes}, plane_size=({H},{W})")

                for idx in range(num_planes):
                    slice_img = xsl[:, idx, :, :].unsqueeze(0).to(model_device)  # (1,C,H,W)
                    out = self._window_inference(slice_img).squeeze()  # (Cout, H, W) or (H, W)
                    if out.dim() == 2:
                        out = out.unsqueeze(0)
                    out_cpu = out.detach().cpu()  # move result to cpu 
                    # #out_cpu = out[-1]
                    # if(p == 0):
                    #     plot_image(out_cpu[-1].cpu().numpy())

                    # assume first two channels are in-plane flows, last channel is prob
                    if out_cpu.shape[0] < 2:
                        raise RuntimeError(f"_window_inference returned unexpected channel count: {out_cpu.shape[0]}")
                    flow0 = out_cpu[0]        # corresponds to slice H axis
                    flow1 = out_cpu[1]        # corresponds to slice W axis
                    prob  = out_cpu[-1]       # last channel as probability (works for 3- or 4-channel outputs)

                    if p == 0:
                        # Z-slice: idx is z, H=X, W=Y  -> add to yf[:, z, :, :]
                        yf[cp[p][0], idx, :, :] += flow0
                        yf[cp[p][1], idx, :, :] += flow1
                        yf[3, idx, :, :]        += prob

                    elif p == 1:
                        # Y-slice: idx is y, H=Z, W=Y -> flow0 maps to Z axis, flow1 to Y axis
                        # flow0 shape: (Z, Y) -> fits yf[:, :, idx, :]
                        yf[cp[p][0], :, idx, :] += flow0
                        yf[cp[p][1], :, idx, :] += flow1
                        yf[3, :, idx, :]        += prob
                    else:  # p == 2
                        # X-slice: idx is x, H=Z, W=X -> flow0 maps to Z axis, flow1 to X axis
                        # flow0 shape: (Z, X) -> fits yf[:, :, :, idx]
                        yf[cp[p][0], :, :, idx] += flow0
                        yf[cp[p][1], :, :, idx] += flow1
                        yf[3, :, :, idx]        += prob

                    #if p == 0:
                        #show_QC_results(slice_img[0,0].cpu().numpy(), yf[3, idx, :, :].cpu().numpy(), yf[3, idx, :, :].cpu().numpy())
                    # elif p == 1:
                    #     show_QC_results(slice_img[0,0].cpu().numpy(),yf[3, :, idx, :].cpu().numpy(), yf[3, :, idx, :].cpu().numpy())
                    # else:  # p == 2
                    #     show_QC_results(slice_img[0,0].cpu().numpy(),yf[3, :, :, idx].cpu().numpy(), yf[3, :, :, idx].cpu().numpy())
                    # free memory 

                    del slice_img, out, out_cpu, flow0, flow1, prob
                    torch.cuda.empty_cache()
                #show_QC_results(slice_img[0,0].cpu().numpy(), yf[-1, 2].cpu().numpy(), yf[-1,2].cpu().numpy())
            
            # show_QC_results(slice_img[0,30].cpu().numpy(), yf[-1, 30].cpu().numpy(), yf[-1,30].cpu().numpy())
        return yf
    
    @torch.no_grad()
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

        # print(f"dP shape (flow maps): {dP.shape}")  
        # print(f"cellprob shape (raw probability map): {cellprob.shape}")  
        # print(f"cellprob after unsqueeze (adding channel dimension): {cellprob.unsqueeze(0).shape}")  
        # print(f"pred_mask shape (combined flow + probability map): {pred_mask.shape}")  

        pred_mask = torch.cat([dP, cellprob.unsqueeze(0)], dim=0)

        pred_mask = pred_mask.cpu().squeeze()
        img_base.cpu()

        return pred_mask

        # if not self.use_tta:
        #     pred_mask = outputs_base
        #     return pred_mask

        # else:
        #     # HorizontalFlip TTA
        #     img_hflip = self.hflip_tta.apply_aug_image(img_data, apply=True)
        #     outputs_hflip = self._window_inference(img_hflip)
        #     outputs_hflip = self.hflip_tta.apply_deaug_mask(outputs_hflip, apply=True)
        #     outputs_hflip = outputs_hflip.cpu().squeeze()
        #     img_hflip = img_hflip.cpu()

        #     # VertricalFlip TTA
        #     img_vflip = self.vflip_tta.apply_aug_image(img_data, apply=True)
        #     outputs_vflip = self._window_inference(img_vflip)
        #     outputs_vflip = self.vflip_tta.apply_deaug_mask(outputs_vflip, apply=True)
        #     outputs_vflip = outputs_vflip.cpu().squeeze()
        #     img_vflip = img_vflip.cpu()

        #     # Merge Results
        #     pred_mask = torch.zeros_like(outputs_base)
        #     pred_mask[0] = (outputs_base[0] + outputs_hflip[0] - outputs_vflip[0]) / 3
        #     pred_mask[1] = (outputs_base[1] - outputs_hflip[1] + outputs_vflip[1]) / 3
        #     pred_mask[2] = (outputs_base[2] + outputs_hflip[2] + outputs_vflip[2]) / 3

        # return pred_mask

    def _post_process3D(self, pred_mask, cellcenters): ## @todo

        """Generate cell instance masks."""
        dP, cellprob = pred_mask[:3], self._sigmoid(pred_mask[-1])
        Z, H, W = pred_mask.shape[-3], pred_mask.shape[-2], pred_mask.shape[-1]
        if hasattr(self, "cellprob_threshold") and self.cellprob_threshold is not None:
            cellprob_threshold = self.cellprob_threshold
        else:
            cellprob_threshold = 0.5
        os.makedirs(self.output_path, exist_ok=True)

        #self.plot_image(cellprob[30])

        if np.prod(H * W) < (5000 * 5000):
            pred_mask = compute_masks3D(
                dP,
                cellprob,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=0.4,
                do_3D=True,
                device=self.device
            )

        else: 
            ##@todo
            print("\n[Whole Slide] Grid Prediction starting...")
            roi_size = 2000

            # Get patch grid by roi_size
            if H % roi_size != 0:
                n_H = H // roi_size + 1
                new_H = roi_size * n_H
            else:
                n_H = H // roi_size
                new_H = H

            if W % roi_size != 0:
                n_W = W // roi_size + 1
                new_W = roi_size * n_W
            else:
                n_W = W // roi_size
                new_W = W

            # Allocate values on the grid
            pred_pad = np.zeros((new_H, new_W), dtype=np.uint32)
            dP_pad = np.zeros((2, new_H, new_W), dtype=np.float32)
            cellprob_pad = np.zeros((new_H, new_W), dtype=np.float32)

            dP_pad[:, :H, :W], cellprob_pad[:H, :W] = dP, cellprob

            for i in range(n_H):
                for j in range(n_W):
                    print("Pred on Grid (%d, %d) processing..." % (i, j))
                    dP_roi = dP_pad[
                        :,
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]
                    cellprob_roi = cellprob_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ]

                    pred_mask = compute_masks(
                        dP_roi,
                        cellprob_roi,
                        use_gpu=True,
                        flow_threshold=0.4,
                        device=self.device,
                        cellprob_threshold=0.5,
                    )[0]

                    pred_pad[
                        roi_size * i : roi_size * (i + 1),
                        roi_size * j : roi_size * (j + 1),
                    ] = pred_mask

            pred_mask = pred_pad[:H, :W]
        if(cellcenters is not None):
            pred_mask = filter_false_positives(pred_mask, cellcenters)
        return pred_mask
    
    


"""
Adapted from the following references:
[1] https://github.com/qubvel/ttach/blob/master/ttach/transforms.py

"""


def hflip(x):
    """flip batch of images horizontally"""
    return x.flip(3)


def vflip(x):
    """flip batch of images vertically"""
    return x.flip(2)


class DualTransform:
    identity_param = None

    def __init__(
        self, name: str, params,
    ):
        self.params = params
        self.pname = name

    def apply_aug_image(self, image, *args, **params):
        raise NotImplementedError

    def apply_deaug_mask(self, mask, *args, **params):
        raise NotImplementedError


class HorizontalFlip(DualTransform):
    """Flip images horizontally (left -> right)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = hflip(image)
        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = hflip(mask)
        return mask


class VerticalFlip(DualTransform):
    """Flip images vertically (up -> down)"""

    identity_param = False

    def __init__(self):
        super().__init__("apply", [False, True])

    def apply_aug_image(self, image, apply=False, **kwargs):
        if apply:
            image = vflip(image)

        return image

    def apply_deaug_mask(self, mask, apply=False, **kwargs):
        if apply:
            mask = vflip(mask)

        return mask
