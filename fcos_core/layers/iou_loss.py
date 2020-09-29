# GIoU and Linear IoU are added by following
# https://github.com/yqyao/FCOS_PLUS/blob/master/maskrcnn_benchmark/layers/iou_loss.py.
import torch
from torch import nn


class IOULoss(nn.Module):
    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        is_3D=False
        if (pred.size(1)==6):
            is_3D=True


        if is_3D:
            pred_left = pred[:, 0]
            pred_top = pred[:, 1]
            pred_front = pred[:, 2]
            pred_right = pred[:, 3]
            pred_bottom = pred[:, 4]
            pred_behind = pred[:, 5]

            target_left = target[:, 0]
            target_top = target[:, 1]
            target_front = target[:, 2]
            target_right = target[:, 3]
            target_bottom = target[:, 4]
            target_behind = target[:, 5]

            target_area = (target_left + target_right) * \
                        (target_top + target_bottom) * \
                        (target_front + target_behind)
            pred_area = (pred_left + pred_right) * \
                        (pred_top + pred_bottom) * \
                        (pred_front + pred_behind)

        else:
            pred_left = pred[:, 0]
            pred_top = pred[:, 1]
            pred_right = pred[:, 2]
            pred_bottom = pred[:, 3]

            target_left = target[:, 0]
            target_top = target[:, 1]
            target_right = target[:, 2]
            target_bottom = target[:, 3]

            target_area = (target_left + target_right) * \
                        (target_top + target_bottom)
            pred_area = (pred_left + pred_right) * \
                        (pred_top + pred_bottom)



        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)

        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        if is_3D:
            d_intersect = torch.min(pred_front, target_front) + torch.min(pred_behind, target_behind)
            g_d_intersect = torch.max(pred_front, target_front) + torch.max(pred_behind, target_behind)
        else:
            d_intersect = 1 # 1 makes no different in area
            g_d_intersect = 1 # 1 makes no different in area
        ac_uion = g_w_intersect * g_h_intersect * g_d_intersect
        area_intersect = w_intersect * h_intersect * d_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect ) / (area_union + 1e-3)
        gious = ious - (ac_uion - area_union) / (ac_uion + 1e-3)
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()
