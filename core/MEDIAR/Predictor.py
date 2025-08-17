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
            roi_size=512,
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
                out = self._window_inference(slice_img).squeeze() #shape (3, HW)
                outputs.append(out) #remove 1st batch dim
                #show_QC_results(slice_img[0,0].cpu().numpy(), out[-1].cpu().numpy(), out[-1].cpu().numpy())


            # Stack outputs along Z
            y = torch.stack(outputs, dim=1)  #shape(4, Z, H, W)

            y_p = y[-1].permute(ipm[p])
            yf[-1] += y_p
            #pltval= self._sigmoid(yf[-1,30,:,:].cpu().squeeze())
            #self.plot_imageSlider(image=pltval)
            for j in range(2):
                yf[cp[p][j]] += y[cpy[p][j]].permute(ipm[p])
            y = None; del y
    
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
        # if(cellcenters is not None):
        #     pred_mask = filter_false_positives(pred_mask, cellcenters)
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
