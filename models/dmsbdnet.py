import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import timm
from .dmscb import DMSCB
from safetensors.torch import load_file
from .sbdm import SBDMDecoder

class MobileNetV3Encoder(nn.Module):
    def __init__(
            self,
            weight_path=None
    ):
        super().__init__()
        self.encoder=timm.create_model(

            "tf_mobilenetv3_large_100",

            pretrained=False,

            features_only=True

        )
        if weight_path is not None:
            checkpoint = load_file(
                weight_path
            )
            msg = self.encoder.load_state_dict(
                checkpoint,
                strict=False
            )
            print(msg)
        self.out_channels=[
            3,
            16,
            24,
            40,
            112,
            960
        ]

    def forward(self,x):

        return self.encoder(x)



class DMSBDNet(nn.Module):

    def __init__(
            self,
            num_classes=6,
            encoder_name="timm-mobilenetv3_large_100",
            pretrained=True
    ):
        super().__init__()
        # self.encoder=smp.encoders.get_encoder(
        #     encoder_name,
        #     weights="imagenet" if pretrained else None
        # )
        self.encoder=MobileNetV3Encoder(weight_path="/mnt/ht2_nas2/EO_test/lzy/DMSBDNet/DMSBDNet-master/models/model.safetensors")
        encoder_channels=self.encoder.out_channels
        # 深层增强
        self.dmscb4=DMSCB(
            encoder_channels[-2]
        )


        self.dmscb5=DMSCB(
            encoder_channels[-1]
        )



        self.decoder=SBDMDecoder(

            encoder_channels,

            decoder_channels=64,

            num_classes=num_classes

        )




    def forward(self,x):


        features=self.encoder(x)



        features[-2]=self.dmscb4(
            features[-2]
        )


        features[-1]=self.dmscb5(
            features[-1]
        )


        seg_out, boundary_out=self.decoder(
            features
        )
        return seg_out, boundary_out