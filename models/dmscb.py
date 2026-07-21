import torch
import torch.nn as nn


class DMSCB(nn.Module):
    """
    Dynamic Multi-scale Separable Convolution Block

    动态多尺度可分离卷积模块

    输入:
        B,C,H,W

    输出:
        B,C,H,W
    """

    def __init__(
            self,
            channels,
            reduction=4
    ):
        super().__init__()
        self.channels = channels
        # 特征压缩
        self.reduce = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        # 多尺度深度卷积分支
        self.dw3 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=5,
                padding=2,
                groups=channels,
                bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.dw7 = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False
            ),

            nn.BatchNorm2d(channels),

            nn.GELU()
        )

        # 动态权重生成
        self.weight_generator = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels // reduction,
                1
            ),
            nn.GELU(),
            nn.Conv2d(
                channels // reduction,
                3,
                1
            )
        )
        # 输出融合
        self.project = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                1,
                bias=False
            ),

            nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()

    def forward(self,x):
        identity=x
        x=self.reduce(x)
        f3=self.dw3(x)
        f5=self.dw5(x)
        f7=self.dw7(x)
        # B,3,1,1
        weight=self.weight_generator(x)

        weight=torch.softmax(
            weight,
            dim=1
        )
        w3=weight[:,0:1]
        w5=weight[:,1:2]
        w7=weight[:,2:3]
        out=(
            w3*f3
            +
            w5*f5
            +
            w7*f7
        )
        out=self.project(out)
        out=out+identity
        return self.act(out)