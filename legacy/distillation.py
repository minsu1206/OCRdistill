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


class KD_MultiBoxLoss(nn.Module):
    """
    Knowledge distillation
    : detection box loss for object detection task.

    Implementation for SSDLite


    Reference:
        https://github.com/SsisyphusTao/Object-Detection-Knowledge-Distillation/blob/dev/odkd/train/loss.py
        detection box : B x N x (x, y, w, h, c) shape.
            B = batch size
            N = number of boxes

    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.negative_ratio = args.negative_ratio
        self.localisation_loss = nn.SmoothL1Loss(reduction='sum')
        self.confidence_loss = nn.CrossEntropyLoss(reduction='sum')
        self.mse_loss = nn.MSELoss()
        self.softmax = nn.Softmax(1)
        self.bound_margin = args.margin
        self.negative_mining = args.OHEM
        self.bound_loss_weight = args.teacher_bound_weight

    def forward(self, student_output, teacher_output, labels):
        """
        Note
            SSDlite output = (B, priors, 4)
            B -> batch size
            priors -> determined by model architecture
            4 -> center x, center y, w, h

        (1) for Ground Truth Label <-> Student model output
            use

        (1) for Teacher model output <-> Student model output

        """

        students_loc = student_output[:, :, :-1]
        students_conf = student_output[:, :, -1]
        teacher_loc = teacher_output[:, :, :-1]
        teacher_conf = teacher_output[:, :, -1]

        # NMS : non - maximum - suppression
        # TODO

        if self.negative_mining:
            # OHEM
            # TODO
            raise NotImplementedError()

        loss_student_gt = self.localisation_loss(students_loc, labels)
        loss_teacher_bound = self.teacher_bounded_loss(teacher_loc, labels)

        return loss_student_gt + self.bound_loss_weight * loss_teacher_bound

    def teacher_bounded_loss(self, student_boxes, teacher_boxes, labels):
        diff_student_gt = self.mse_loss(student_boxes, labels)
        diff_teacher_gt = self.mse_loss(teacher_boxes, labels)

        if diff_student_gt + self.bound_margin > diff_teacher_gt:
            return diff_student_gt
        else:
            return 0







