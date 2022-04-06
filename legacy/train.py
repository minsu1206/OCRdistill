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


def train():
    args = get_parse()

    # Model
    # TODO (*) : model build functions
    Models = model.ModelBuild(args)
    teacher = Models.teacher_model
    student = Models.student_model

    # Dataset
    # TODO : dataset 추가 (ICDAR 2017, COCO text, 등)
    train_dataset = dataload.ICDAR_Dataset(img_dir=args.dataset_path,
                                           annotation_dir=args.annotation_path,
                                           transform=dataload.Augment(img_size=args.img_size))

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_fn)

    # Learning
    optimizer = torch.optim.Adam(student.parameters(),
                                 lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                  lr_lambda=lambda epoch: 0.95 ** epoch)

    # KD loss
    # TODO (*) : Knowledge Distillation loss implementation
    # [V] : KD divergence loss - NIPS 2014
    # [ ] : detection box loss - NIPS 2017
    # [ ] : feature imitation loss - ICLR 2015

    # TODO : prediction result = detection box
    for i, (img, label) in tqdm(enumerate(train_dataloader)):
        student_pred = student(img)
        with torch.no_grad():   # teacher model : parameter freeze
            teacher_pred = teacher(img)

        """
            student_pred = detection box, (B, N, 4 or 5, 8 or 9) torch tensor
                            : x, y, w, h, conf(score)
                            : 
            teacher_pred = detection box, (B, M, 4 or 5, 8 or 9 ) torch tensor
                            : x, y, w, h, conf(score)
        """








if __name__ == "__main__":
    train()

