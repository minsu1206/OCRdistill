
import torch
import numpy as np
import torch.nn as nn
from utils.general import non_max_suppression

def make_mask(imgs, targets):
    """
        imgs = (B, C, H, W )
        targets = ((#Label of B1, #Label of B2, ...), 6)
            6 -> [batch index, cls=0, scaled x ,scaled y, scaled w, scaled h]
    """
    B, C, H, W = imgs.shape
    masks = torch.zeros((B, H, W))
    for target in targets:
        batch_index = target[0].long()
        scaled_x = target[2]
        scaled_y = target[3]
        scaled_w = target[4]
        scaled_h = target[5]

        x = int(scaled_x * W)
        w = int(scaled_w * W)
        y = int(scaled_y * H)
        h = int(scaled_h * H)
        masks[batch_index, x:x+w, y:y+h] = 1

    return masks



def element_count(gt_masks, teacher_masks):
    # print(gt_masks.shape) # (B, 640, 640)
    batch = gt_masks.shape[0]
    ratios_bg = torch.zeros((batch, 1))
    ratios_obj = torch.zeros((batch, 1))
    original_gt_area = torch.sum(gt_masks.reshape(batch, -1), dim=-1)
    original_teacher_area = torch.sum(teacher_masks.reshape(batch, -1), dim=-1)
    original_gt_area[original_gt_area <= 0] = original_gt_area.mean()
    gt_minus_teacher = gt_masks - teacher_masks
    gt_minus_teacher[gt_minus_teacher >= 0] = 0
    gt_minus_teacher[gt_minus_teacher < 0] = 1
    gt_minus_teacher_area = torch.sum(gt_minus_teacher.reshape(batch, -1), dim=-1)
    # 배경인데 Teacher 가 잘못 예측한 경우

    # print(gt_minus_teacher_area)
    teacher_minus_gt = teacher_masks - gt_masks
    teacher_minus_gt[teacher_minus_gt >= 0] = 0
    teacher_minus_gt[teacher_minus_gt < 0] = 1
    teacher_minus_gt_area = torch.sum(teacher_minus_gt.reshape(batch, -1), dim=-1)
    # GT 인데 Teacher 가 예측을 안한 경우

    for b in range(batch):
        ratio_bg = gt_minus_teacher_area[b] / original_gt_area[b]
        ratio_obj = (original_gt_area[b] +  teacher_minus_gt_area[b]) / original_gt_area[b]
        # print(b , ": ", ratio_bg, ratio_obj)
        ratios_bg[b] = ratio_bg
        ratios_obj[b] = ratio_obj
    # 잘 못 예측한 output 들의 영역들이 많이 남음

    return ratios_bg, ratios_obj

