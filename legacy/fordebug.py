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
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train():
    args = get_parse()

    # Model
    # TODO (*) : model build functions
    Models = model.ModelBuild(args)
    # teacher = Models.teacher_model
    student_config = Models.student_model()
    student_model = student_config['MODEL']
    student_model.cuda()
    student_imgsize = student_config['IMGSIZE']

    # Dataset
    # TODO : dataset 추가 (ICDAR 2017, COCO text, 등)
    train_dataset = dataload.ICDAR_Dataset(img_dir=args.dataset_path,
                                           annotation_dir=args.annotation_path,
                                           transform=dataload.Augment(img_size=args.img_size))

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

    for i, (data, label) in enumerate(train_dataloader):
        data = data.cuda()
        print(data.shape)
        output = student_model(data)
        print(output.shape)
        exit()


if __name__ == "__main__":
    train()
