import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import csv
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from models.dmsbdnet import DMSBDNet
from datasets.potsdam_dataset_class5 import PotsdamDatasetClass5
from Losses.loss import DMSBDLoss5

import albumentations as A
import random
import numpy as np


# ======================
# 参数
# ======================

IMAGE_DIR = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/potsdam"

TRAIN_IMAGE_DIR = os.path.join(
    IMAGE_DIR,
    "img_dir",
    "train"
)

TRAIN_MASK_DIR = os.path.join(
    IMAGE_DIR,
    "ann_dir",
    "train"
)


TEST_IMAGE_DIR = os.path.join(
    IMAGE_DIR,
    "test",
    "img_dir"
)

TEST_MASK_DIR = os.path.join(
    IMAGE_DIR,
    "test",
    "ann_dir"
)

CLASS_NAMES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car"
]
NUM_CLASSES = 5
IGNORE_LABEL = 5

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
CROP_SIZE = 512
SAVE_DIR = "./checkpoints0720_class5"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================
# 数据增强
# ======================
train_transform = A.Compose([
    A.RandomCrop(height=CROP_SIZE, width=CROP_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Affine(translate_percent=0.1, scale=(0.9,1.1), rotate=(-30,30), p=0.5)
])

test_transform = A.Compose([])


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed: {seed}")


# ======================
# 指标计算：IoU / Precision / Recall / F1
# ======================
def compute_metrics(pred, mask, num_classes):
    pred = torch.argmax(pred, dim=1)
    # 过滤忽略区域
    valid_mask = mask != IGNORE_LABEL
    pred_valid = pred[valid_mask]
    mask_valid = mask[valid_mask]

    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for cls in range(num_classes):
        pred_cls = (pred_valid == cls)
        mask_cls = (mask_valid == cls)

        tp = (pred_cls & mask_cls).sum().float()
        fp = (pred_cls & ~mask_cls).sum().float()
        fn = (~pred_cls & mask_cls).sum().float()

        # IoU
        union = tp + fp + fn
        iou = (tp / union).item() if union > 0 else 0.0

        # Precision / Recall / F1
        prec = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        iou_list.append(iou)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

    miou = sum(iou_list) / num_classes
    mf1 = sum(f1_list) / num_classes

    return miou, mf1, iou_list, f1_list


# ======================
# Train one epoch
# ======================
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0

    batch_mious = []
    batch_mf1s = []
    batch_ious = []
    batch_f1s = []

    for image, mask in loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            pred, boundary_pred = model(image)
            loss = criterion(pred, boundary_pred, mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        miou, mf1, iou_list, f1_list = compute_metrics(pred.detach(), mask, NUM_CLASSES)
        batch_mious.append(miou)
        batch_mf1s.append(mf1)
        batch_ious.append(iou_list)
        batch_f1s.append(f1_list)

    # 逐类别平均
    epoch_ious = [sum(x)/len(x) for x in zip(*batch_ious)]
    epoch_f1s = [sum(x)/len(x) for x in zip(*batch_f1s)]
    epoch_miou = sum(epoch_ious) / NUM_CLASSES
    epoch_mf1 = sum(epoch_f1s) / NUM_CLASSES

    return total_loss / len(loader), epoch_miou, epoch_mf1, epoch_ious, epoch_f1s


# ======================
# Test
# ======================
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    batch_mious = []
    batch_mf1s = []
    batch_ious = []
    batch_f1s = []

    for image, mask in loader:
        image = image.to(DEVICE)
        mask = mask.to(DEVICE)
        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            pred, boundary_pred = model(image)
            loss = criterion(pred, boundary_pred, mask)

        total_loss += loss.item()
        miou, mf1, iou_list, f1_list = compute_metrics(pred, mask, NUM_CLASSES)
        batch_mious.append(miou)
        batch_mf1s.append(mf1)
        batch_ious.append(iou_list)
        batch_f1s.append(f1_list)

    epoch_ious = [sum(x)/len(x) for x in zip(*batch_ious)]
    epoch_f1s = [sum(x)/len(x) for x in zip(*batch_f1s)]
    epoch_miou = sum(epoch_ious) / NUM_CLASSES
    epoch_mf1 = sum(epoch_f1s) / NUM_CLASSES

    return total_loss / len(loader), epoch_miou, epoch_mf1, epoch_ious, epoch_f1s


# ======================
# Train
# ======================
def train():
    train_dataset = PotsdamDatasetClass5(TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, train_transform)
    test_dataset = PotsdamDatasetClass5(TEST_IMAGE_DIR, TEST_MASK_DIR, test_transform)
    print(len(train_dataset))
    print(len(test_dataset))
    print(train_dataset.images[:10])
    print(test_dataset.images[:10])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = DMSBDNet(num_classes=NUM_CLASSES)
    model.to(DEVICE)
    criterion = DMSBDLoss5()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_miou = 0
    os.makedirs(SAVE_DIR, exist_ok=True)
    log_file = os.path.join(SAVE_DIR, "train_log.csv")

    # 完整表头：IoU + F1
    header = [
        "epoch",
        "train_loss", "train_miou", "train_mf1",
        "train_iou_impervious", "train_iou_building", "train_iou_lowveg", "train_iou_tree", "train_iou_car",
        "train_f1_impervious", "train_f1_building", "train_f1_lowveg", "train_f1_tree", "train_f1_car",

        "test_loss", "test_miou", "test_mf1",
        "test_iou_impervious", "test_iou_building", "test_iou_lowveg", "test_iou_tree", "test_iou_car",
        "test_f1_impervious", "test_f1_building", "test_f1_lowveg", "test_f1_tree", "test_f1_car",

        "lr"
    ]
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for epoch in range(EPOCHS):
        train_loss, train_miou, train_mf1, train_ious, train_f1s = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        test_loss, test_miou, test_mf1, test_ious, test_f1s = evaluate(model, test_loader, criterion)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # 打印
        print(f"\n========== Epoch [{epoch+1}/{EPOCHS}] ==========")
        print(f"Train Loss:{train_loss:.4f} | mIoU:{train_miou:.4f} | mF1:{train_mf1:.4f}")
        print(f"Test  Loss:{test_loss:.4f} | mIoU:{test_miou:.4f} | mF1:{test_mf1:.4f}")

        print("\n[Train Per-Class IoU / F1]")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"{name:20s} | IoU: {train_ious[idx]:.4f} | F1: {train_f1s[idx]:.4f}")

        print("\n[Test Per-Class IoU / F1]")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"{name:20s} | IoU: {test_ious[idx]:.4f} | F1: {test_f1s[idx]:.4f}")

        # 写入CSV
        row = [
            epoch+1,
            train_loss, train_miou, train_mf1,
            *train_ious, *train_f1s,
            test_loss, test_miou, test_mf1,
            *test_ious, *test_f1s,
            lr
        ]
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # 保存权重
        epoch_weight_path = os.path.join(
            SAVE_DIR,
            f"epoch_{epoch+1:03d}_trainLoss_{train_loss:.4f}_testLoss_{test_loss:.4f}_testmIoU_{test_miou:.4f}.pth"
        )
        torch.save(model.state_dict(), epoch_weight_path)

        if test_miou > best_miou:
            best_miou = test_miou
            best_weight_path = os.path.join(SAVE_DIR, "best_DMSBDNet.pth")
            torch.save(model.state_dict(), best_weight_path)
            print(f"\n✅ Save Best Model | Best mIoU: {best_miou:.4f}")


if __name__=="__main__":
    set_seed(42)
    train()