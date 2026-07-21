import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.GELU(),


            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.GELU()
        )


    def forward(self,x):

        return self.block(x)



class SBDMDecoder(nn.Module):


    def __init__(
            self,
            encoder_channels,
            decoder_channels=64,
            num_classes=6
    ):

        super().__init__()


        self.conv5=ConvBlock(
            encoder_channels[-1],
            decoder_channels
        )


        self.conv4=ConvBlock(
            decoder_channels+encoder_channels[-2],
            decoder_channels
        )


        self.conv3=ConvBlock(
            decoder_channels+encoder_channels[-3],
            decoder_channels
        )


        self.conv2=ConvBlock(
            decoder_channels+encoder_channels[-4],
            decoder_channels
        )


        # 继续恢复分辨率

        self.conv1=ConvBlock(
            decoder_channels,
            decoder_channels
        )


        self.conv0=ConvBlock(
            decoder_channels,
            decoder_channels
        )



        # semantic branch

        self.semantic=nn.Sequential(

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                3,
                padding=1
            ),

            nn.BatchNorm2d(decoder_channels),

            nn.GELU()

        )


        # boundary branch

        self.boundary=nn.Sequential(

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                3,
                padding=1,
                groups=decoder_channels
            ),

            nn.Conv2d(
                decoder_channels,
                decoder_channels,
                1
            ),

            nn.Sigmoid()
        )


        self.fusion=nn.Conv2d(
            decoder_channels,
            decoder_channels,
            1
        )


        self.head=nn.Conv2d(
            decoder_channels,
            num_classes,
            1
        )

        self.boundary_head = nn.Conv2d(
            decoder_channels,
            1,
            1
        )
    def up(
            self,
            x,
            next_feature,
    ):

        return F.interpolate(
            x,
            size=next_feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


    def forward(self,features):


        f1,f2,f3,f4,f5=features[-5:]


        x=self.conv5(f5)


        x=self.up(x,f4)


        x=torch.cat(
            [
                x,
                f4
            ],
            dim=1
        )

        x=self.conv4(x)



        x=self.up(x,f3)

        x=torch.cat(
            [
                x,
                f3
            ],
            dim=1
        )

        x=self.conv3(x)



        x=self.up(x,f2)

        x=torch.cat(
            [
                x,
                f2
            ],
            dim=1
        )

        x=self.conv2(x)



        # 1/4 -> 1/2

        # x=self.up(x)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        x=self.conv1(x)



        # 1/2 -> 1

        # x=self.up(x)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False
        )
        x=self.conv0(x)



        # semantic

        fs=self.semantic(x)


        # boundary attention
        fb=self.boundary(x)
        x=fs*fb
        x=self.fusion(x)
        seg_out=self.head(x)
        boundary_out = self.boundary_head(x)

        return seg_out, boundary_out