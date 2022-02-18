"""
model build

    Teacher model
        1. CharNet
        2.

    Student model
        1. YOLOv3
        2. SSDLite

"""
import torch
import numpy as np
from utils import *


class ModelBuild:
    def __init__(self, args):
        self.args = args

    def teacher_model(self):
        name = self.args.T_model
        model_config = {}

        # TODO
        if name == 'CharNet':
            from charnet.modeling.model import CharNet
            from charnet.modeling.utils import rotate_rect
            from charnet.modeling.rotated_nms import nms
            from charnet.config import cfg
            import torch.nn as nn
            # cfg.merge_from_file(r'C:\Users\user\PycharmProjects\dino_lib22\OCRdistill\research-charnet\configs\icdar2015_hourglass88.yaml')

            model = CharNet()
            model.load_state_dict(torch.load(cfg.WEIGHT))
            model.eval()

            class CharNetCustom(nn.Module):
                def __init__(self):
                    super(CharNetCustom, self).__init__()
                    self.backbone = model.backbone
                    self.detector = model.word_detector

                def forward(self, x, scale_w, scale_h, ori_im_w, ori_im_h):
                    x = torch.unsqueeze(x, dim=0)
                    pred_word_fg, pred_word_tblr, pred_word_orient = self.detector(self.backbone(x))
                    pred_word_fg, pred_word_tblr, pred_word_orient = to_numpy_or_none(
                        pred_word_fg, pred_word_tblr, pred_word_orient
                    )
                    ss_word_bboxes = self.parse_word_bboxes(pred_word_fg[0, 1], pred_word_tblr[0], pred_word_orient[0, 0],
                                                            scale_w, scale_h, ori_im_w, ori_im_h)
                    return ss_word_bboxes

                def parse_word_bboxes(
                        self, pred_word_fg, pred_word_tblr,
                        pred_word_orient, scale_w, scale_h,
                        W, H
                ):
                    word_stride = 4     # WORD_STRIDE = 4
                    word_keep_rows, word_keep_cols = np.where(pred_word_fg > 0.95)
                    # WORD_MIN_SCORE = 0.95
                    oriented_word_bboxes = np.zeros((word_keep_rows.shape[0], 9), dtype=np.float32)
                    for idx in range(oriented_word_bboxes.shape[0]):
                        y, x = word_keep_rows[idx], word_keep_cols[idx]
                        t, b, l, r = pred_word_tblr[:, y, x]
                        o = pred_word_orient[y, x]
                        score = pred_word_fg[y, x]
                        four_points = rotate_rect(
                            scale_w * word_stride * (x - l), scale_h * word_stride * (y - t),
                            scale_w * word_stride * (x + r), scale_h * word_stride * (y + b),
                            o, scale_w * word_stride * x, scale_h * word_stride * y)
                        oriented_word_bboxes[idx, :8] = np.array(four_points, dtype=np.float32).flat
                        oriented_word_bboxes[idx, 8] = score
                    keep, oriented_word_bboxes = nms(oriented_word_bboxes, 0.15, num_neig=1)
                    # WORDS_NMS_IOU_THRESH = 0.15
                    oriented_word_bboxes = oriented_word_bboxes[keep]
                    oriented_word_bboxes[:, :8] = oriented_word_bboxes[:, :8].round()
                    oriented_word_bboxes[:, 0:8:2] = np.maximum(0, np.minimum(W - 1, oriented_word_bboxes[:, 0:8:2]))
                    oriented_word_bboxes[:, 1:8:2] = np.maximum(0, np.minimum(H - 1, oriented_word_bboxes[:, 1:8:2]))
                    return oriented_word_bboxes

            model_config['MODEL'] = CharNetCustom()
            model_config['IMGSIZE'] = cfg.INPUT_SIZE
            return model_config
            # raise NotImplementedError()
            # return
        else:
            raise NotImplementedError()

    def student_model(self):
        model_config = {}
        name = self.args.S_model

        # TODO
        if name == 'yolov3':
            from PyTorch_YOLOv3.models.yolov3 import YOLOv3
            import yaml
            with open('PyTorch_YOLOv3/config/yolov3_default.cfg', 'r') as f:
                cfg = yaml.safe_load(f)
            model = YOLOv3(cfg['MODEL'], ignore_thre=cfg['TRAIN']['IGNORETHRE'])
            data_size = cfg['TRAIN']['IMGSIZE']
            model_config['MODEL'] = model
            model_config['IMGSIZE'] = data_size

        elif name == 'SSDLite':
            # TODO : 용준, 찬미
            # TODO : SSDLite 연결. model_config 에는 위 참고해서 model이랑 img size 넣기
            raise NotImplementedError()
        else:
            raise NotImplementedError()

        return model_config





