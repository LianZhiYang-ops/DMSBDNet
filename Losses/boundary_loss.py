import torch
import torch.nn as nn
import torch.nn.functional as F



class BoundaryLoss(nn.Module):


    def __init__(self):

        super().__init__()
        self.loss=nn.BCEWithLogitsLoss()

    def forward(self,pred_boundary,target):

        """
        pred_boundary:
            B,1,H,W
            logits


        target:
            B,H,W
        """
        gt_boundary=self.extract_boundary(target)
        loss=self.loss(
            pred_boundary,
            gt_boundary
        )
        return loss

    def extract_boundary(self,mask):
        mask=mask.float()
        mask=mask.unsqueeze(1)
        # print("mask shape:", mask.shape)
        # print("mask min:", mask.min().item(), "mask max:", mask.max().item())
        # print("unique values:", torch.unique(mask))
        max_pool=F.max_pool2d(   mask,    3,   1,  1 )
        min_pool=-F.max_pool2d(   -mask,    3,      1,  1 )
        boundary=max_pool-min_pool
        boundary=torch.clamp(boundary,0,1  )
        return boundary