import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import time
from tqdm import tqdm

# custom modules
import dataload
from config import get_parse
import visualize
import model
import distillation
from utils import *
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train():
    args = get_parse()
    use_gpu = torch.cuda.is_available()

    # Model
    # TODO (*) : model build functions
    Models = model.ModelBuild(args)
    teacher_config = Models.teacher_model()
    teacher_model = teacher_config['MODEL']

    teacher_imgsize = teacher_config['IMGSIZE']
    student_config = Models.student_model()
    student_model = student_config['MODEL']

    if use_gpu:
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()

    student_imgsize = student_config['IMGSIZE']
    print(teacher_imgsize, student_imgsize)
    # Dataset
    # TODO : dataset 추가 (ICDAR 2017, COCO text, 등)
    train_dataset = dataload.ICDAR_Dataset(img_dir=args.dataset_path,
                                           annotation_dir=args.annotation_path,
                                           transform=dataload.Augment(teacher_image_size=teacher_imgsize,
                                                                      student_image_size=student_imgsize))

    # TODO : student model 이랑 teacher model 입력 이미지 사이즈가 달라서 rescale_coords가 필요할것 같다... 귀찮구만

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    # Learning
    optimizer = torch.optim.Adam(student_model.parameters(),
                                 lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: 0.95 ** epoch)

    for i, data_bundle in enumerate(train_dataloader):
        teacher_imgs = data_bundle[0]        # List (CharNet)
        teacher_label = data_bundle[1]      # NO use
        student_img = data_bundle[2]        # torch.tensor (YOLOv3)
        student_label = data_bundle[3]
        charnet_scale_infos = data_bundle[4]

        with torch.no_grad():
            teacher_outputs = []
            for teacher_img, scale_info in zip(teacher_imgs, charnet_scale_infos):
                if use_gpu:
                    teacher_img = teacher_img.cuda()
                charnet_in = [teacher_img] + scale_info
                teacher_output = teacher_model(charnet_in[0], charnet_in[1], charnet_in[2], charnet_in[3], charnet_in[4])
                print('T', teacher_output)
                teacher_outputs.append(teacher_output)

        if use_gpu:
            student_img = student_img.cuda()
        student_output = student_model(student_img)
        print('S', student_output.shape)


if __name__ == "__main__":
    train()
