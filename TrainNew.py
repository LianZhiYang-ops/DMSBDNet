import os
import csv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.dmsbdnet import DMSBDNet
from datasets.potsdam_dataset import PotsdamDataset
from Losses.loss import DMSBDLoss

import albumentations as A
import random
import numpy as np


# ======================
# 参数
# ======================

IMAGE_DIR = r"D:\code\CropSeg\archive\Pots"

TRAIN_IMAGE_DIR = os.path.join(
    IMAGE_DIR,
    "train",
    "rgb"
)

TRAIN_MASK_DIR = os.path.join(
    IMAGE_DIR,
    "train",
    "label"
)


TEST_IMAGE_DIR = os.path.join(
    IMAGE_DIR,
    "test",
    "rgb"
)

TEST_MASK_DIR = os.path.join(
    IMAGE_DIR,
    "test",
    "label"
)



NUM_CLASSES = 6

BATCH_SIZE = 4

EPOCHS = 100

LR = 1e-4



DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ======================
# 数据增强
# ======================


train_transform = A.Compose([

    A.HorizontalFlip(
        p=0.5
    ),

    A.VerticalFlip(
        p=0.5
    ),

    A.RandomRotate90(
        p=0.5
    ),


    A.RandomBrightnessContrast(
        p=0.3
    ),


    A.Affine(
        translate_percent=0.1,
        scale=(0.9,1.1),
        rotate=(-30,30),
        p=0.5
    )

])



test_transform = A.Compose([])


def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)


    # 保证确定性
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


    print(
        f"Random seed: {seed}"
    )
# ======================
# mIoU
# ======================


def compute_miou(
        pred,
        mask,
        num_classes
):
    pred=torch.argmax(pred,dim=1)
    # print(torch.equal(pred, mask))
    # print(torch.unique(pred))
    # print(torch.unique(mask))
    miou=[]
    for cls in range(num_classes):
        pred_cls = (pred==cls)
        mask_cls = (mask==cls)
        intersection=(
            pred_cls &
            mask_cls
        ).sum().item()
        union=(pred_cls |mask_cls).sum().item()
        if union!=0:
            miou.append(intersection/union)
    if len(miou)==0:
        return 0
    return sum(miou)/len(miou)

# ======================
# Train one epoch
# ======================

def train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        scaler
):
    model.train()
    total_loss=0
    total_miou=0
    for image,mask in loader:
        image=image.to(
            DEVICE
        )
        mask=mask.to(
            DEVICE
        )
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            pred,boundary_pred=model(
                image
            )
            loss=criterion(
                pred,
                boundary_pred,
                mask
            )
        scaler.scale(
            loss
        ).backward()
        scaler.step(
            optimizer
        )
        scaler.update()
        total_loss += loss.item()
        total_miou += compute_miou(
            pred.detach(),
            mask,
            NUM_CLASSES
        )
    return (
        total_loss/len(loader),
        total_miou/len(loader)
    )



# ======================
# Test
# ======================


@torch.no_grad()
def evaluate(
        model,
        loader,
        criterion
):
    model.eval()
    total_loss=0
    total_miou=0

    for image,mask in loader:
        image=image.to(
            DEVICE
        )
        mask=mask.to(
            DEVICE
        )
        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):
            pred,boundary_pred=model(
                image
            )
            loss=criterion(
                pred,
                boundary_pred,
                mask
            )
        total_loss+=loss.item()
        total_miou+=compute_miou(
            pred,
            mask,
            NUM_CLASSES
        )
    return (
        total_loss/len(loader),
        total_miou/len(loader)
    )



# ======================
# Train
# ======================


def train():

    train_dataset=PotsdamDataset(

        TRAIN_IMAGE_DIR,

        TRAIN_MASK_DIR,

        train_transform

    )

    test_dataset=PotsdamDataset(

        TEST_IMAGE_DIR,

        TEST_MASK_DIR,

        test_transform

    )
    print(len(train_dataset))
    print(len(test_dataset))

    print(train_dataset.images[:10])
    print(test_dataset.images[:10])
    # exit(0)
    train_loader=DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    test_loader=DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True

    )

    model=DMSBDNet(
        num_classes=NUM_CLASSES
    )
    model.to(
        DEVICE
    )
    criterion=DMSBDLoss()
    optimizer=torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4

    )
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS

    )
    scaler=torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )
    best_miou=0
    # 保存日志
    log_file="train_log.csv"
    with open(
        log_file,
        "w",
        newline=""
    ) as f:
        writer=csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_miou",
            "test_loss",
            "test_miou",
            "lr"
        ])
    for epoch in range(EPOCHS):
        train_loss,train_miou=train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )
        test_loss,test_miou=evaluate(
            model,
            test_loader,
            criterion
        )
        scheduler.step()
        lr=optimizer.param_groups[0]["lr"]
        print(

            f"Epoch [{epoch+1}/{EPOCHS}] "

            f"Train Loss:{train_loss:.4f} "

            f"Train mIoU:{train_miou:.4f} "

            f"Test Loss:{test_loss:.4f} "

            f"Test mIoU:{test_miou:.4f}"

        )



        with open(
            log_file,
            "a",
            newline=""
        ) as f:


            writer=csv.writer(f)


            writer.writerow([

                epoch+1,

                train_loss,

                train_miou,

                test_loss,

                test_miou,

                lr

            ])




        if test_miou > best_miou:


            best_miou=test_miou


            torch.save(

                model.state_dict(),

                "best_DMSBDNet.pth"

            )


            print(
                "Save best model"
            )



if __name__=="__main__":
    set_seed(42)
    train()