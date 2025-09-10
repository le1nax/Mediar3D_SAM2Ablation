import torch
import torch.nn as nn

from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.base.modules import Activation

from .CustomLightWeightDecoder import CustomLeightWeightDecoder
from .CustomDecoder import CustomMAnetDecoder
from .HieraEncoder import HieraEncoderWrapper

__all__ = ["MEDIARFormer"]

def convert_bn_to_float32(module):
    if isinstance(module, nn.BatchNorm2d):
        module.float()
    for child in module.children():
        convert_bn_to_float32(child)

class MEDIARFormer(MAnet):
    """MEDIAR-Former Model"""

    def __init__(
        self,
        encoder_name="mit_b4",  # Default encoder
        encoder_weights=None,  # Pre-trained weights
        encoder_channels=(256, 256, 256, 256),  # encoder configuration #if sclap=0 adapt to 4 dims
        decoder_channels=(256, 256, 256, 256, 256),  # Decoder configuration
        decoder_pab_channels=256,  # Decoder Pyramid Attention Block channels
        in_channels=3,  # Number of input channels
        classes=3,  # Number of output classes
    ):
         # init MAnet with dummy encoder (will be replaced)
        super().__init__(
            encoder_name="mit_b4",
            encoder_weights=None,
            encoder_depth=5,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels,
            in_channels=in_channels,
            classes=classes,
        )

        # swap SegFormer encoder for Hiera
        self.encoder = HieraEncoderWrapper(hiera_cfg="sam2_hiera_l.yaml")


        self.decoder = CustomMAnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,  # Use batch normalization in decoder
            pab_channels=decoder_pab_channels,
        )

        convert_bn_to_float32(self.decoder)

        # remove segmentation head
        self.segmentation_head = None

        # convert activations
        _convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
        _convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))

        # custom heads
        self.cellprob_head = DeepSegmentationHead(in_channels=decoder_channels[-1], out_channels=1, upsampling=2)
        self.gradflow_head = DeepSegmentationHead(in_channels=decoder_channels[-1], out_channels=2, upsampling=2)


    #Override the check_input_shape method to ensure input shape is compatible with the encoder
    def check_input_shape(self, x):
        # Ensure input has correct channel count for the encoder
        expected_in_ch = getattr(self.encoder, "in_channels", 3)  # default 3 if not defined
        if x.shape[1] != expected_in_ch:
            if x.shape[1] == 1 and expected_in_ch == 3:
                # Repeat the single channel to make it 3 channels
                x = x.repeat(1, 3, 1, 1)
            elif x.shape[1] == 4 and expected_in_ch == 3:
                # Use only the first 3 channels
                x = x[:, :3, :, :]

            else:
                raise RuntimeError(
                    f"Input has {x.shape[1]} channels, but encoder expects {expected_in_ch}."
                )

        # Check spatial dimensions
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )
        
        return x

    def forward(self, x):
        """Forward pass through the network"""
        # Ensure the input shape is correct
        x = self.check_input_shape(x)

        # Encode the input and then decode it
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        # Generate masks for cell probability and gradient flows
        cellprob_mask = self.cellprob_head(decoder_output)
        gradflow_mask = self.gradflow_head(decoder_output)

        # Concatenate the masks for output
        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks


class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        # Define a sequence of layers for the segmentation head
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)
