import torch.nn as nn
import torch
import torch.nn.functional as F


#
# def KD_loss_build(methods: list):
#     losses = []
#     for method in methods:
#         if method == 0:
#             losses.append(KD_KLdiv())
#         else:
#             pass
#     raise NotImplementedError()


def KD_KLdiv(student_output, teacher_output, label, params):
    """
    KD divergence loss btw teacher's soft label and student's soft label
    for classification task
    """
    alpha = params.alpha
    T = params.temperature

    loss_btw_student_teacher = nn.KLDivLoss()(
        F.log_softmax(student_output / T, dim=1),
        F.softmax(teacher_output / T, dim=1)) * (alpha * T * T)

    loss_btw_student_gt = F.cross_entropy(student_output, label) * (1 - alpha)

    return loss_btw_student_gt + loss_btw_student_teacher

    # raise NotImplementedError()


def KD_det_box(student_output, teacher_output, label):
    """
    KD detection box loss for object detection task.

    Reference:
        https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation/blob/dev/odkd/train/loss.py
    """
    # TODO
    localisation_loss = nn.SmoothL1Loss(reduction='sum')
    confidence_loss = nn.CrossEntropyLoss(reduction='sum')
