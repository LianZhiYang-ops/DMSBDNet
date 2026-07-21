import torch.nn.functional as F

from .boundary_loss import BoundaryLoss
import torch
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=5):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        # logits: [B, num_classes, H, W] 模型原始输出
        # target: [B, H, W] 类别标签
        B, C, H, W = logits.shape
        pred = F.softmax(logits, dim=1)

        # 1. 生成掩码，过滤ignore_index=5的像素
        valid_mask = (target != self.ignore_index)
        # 只保留有效像素，忽略标签5
        target_valid = target.clone()
        target_valid[~valid_mask] = 0  # 临时填充，onehot不会用到这部分

        # 2. one-hot编码
        one_hot_target = F.one_hot(target_valid, num_classes=C).permute(0, 3, 1, 2)
        # 拓展valid_mask到通道维度 [B,1,H,W] -> [B,C,H,W]
        valid_mask_exp = valid_mask.unsqueeze(1).expand(-1, C, -1, -1)

        # 3. 仅计算有效像素的交并，忽略ignore区域
        pred = pred * valid_mask_exp
        one_hot_target = one_hot_target * valid_mask_exp

        intersection = torch.sum(pred * one_hot_target, dim=(2, 3))
        pred_sum = torch.sum(pred, dim=(2, 3))
        target_sum = torch.sum(one_hot_target, dim=(2, 3))

        dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        dice_loss = 1 - torch.mean(dice_coeff)
        return dice_loss


class DMSBDLoss:
    def __init__(self):
        self.boundary=BoundaryLoss()
    def __call__(
            self,
            sec_pred,
            boundary_pred,
            mask
    ):
        ce=F.cross_entropy(
            sec_pred,
            mask
        )
        boundary=self.boundary(
            boundary_pred,
            mask
        )
        loss=ce+0.2*boundary
        return loss
    

class DMSBDLoss5:
    def __init__(self):
        self.boundary=BoundaryLoss()
    def __call__(
            self,
            sec_pred,
            boundary_pred,
            mask
    ):
        ce=F.cross_entropy(
            sec_pred,
            mask,
            ignore_index=5
        )
        boundary=self.boundary(
            boundary_pred,
            mask
        )
        loss=ce+0.2*boundary
        return loss
    

class DMSBDLoss5_Dice:
    def __init__(self, alpha=0.5, lam=0.2):
        """
        :param alpha: Dice Loss权重
        :param lam: 边界BCE损失权重
        总损失公式：L = CE + α * DiceLoss + λ * BoundaryLoss
        """
        self.boundary = BoundaryLoss()
        self.dice = DiceLoss()
        self.alpha = alpha    # Dice权重，推荐0.4~0.6
        self.lam = lam        # 边界损失权重，推荐0.2~0.35

    def __call__(
            self,
            sec_pred,
            boundary_pred,
            mask
    ):
        # 1. 交叉熵损失
        ce = F.cross_entropy(
            sec_pred,
            mask,
            ignore_index=5
        )
        # 2. Dice损失
        dice = self.dice(sec_pred, mask)
        # 3. 边界辅助损失
        boundary_loss = self.boundary(boundary_pred, mask)
        # 总损失加权求和
        total_loss = ce + self.alpha * dice + self.lam * boundary_loss
        return total_loss
    