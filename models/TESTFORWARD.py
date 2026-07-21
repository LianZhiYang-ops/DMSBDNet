from models.dmsbdnet import DMSBDNet
import torch


model=DMSBDNet(
    num_classes=6
)


x=torch.randn(
    2,
    3,
    512,
    512
)


y=model(x)


print(y.shape)