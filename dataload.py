"""
ICDAR 2015 - Text Localisation task dataset
: 1000 images, 1000 annotation files

Code Reference :
    (1) https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    (2) https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html

Requirements:
    - torch
    - torchvision
    - imgaug

    by MS.KIM
"""

import enum
import torch
import os
import torchvision
from torch.utils.data import Dataset
# from torchvision.io import read_image
import cv2
import torchvision.transforms as transforms
import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


class Augment:
    def __init__(self, teacher_image_size, student_image_size, to_tensor=True):
        self.seq = iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Affine(rotate=(-15, 15)),
            iaa.Affine(shear=(-15, 15)),
        ])
        self.teacher_image_size = teacher_image_size
        self.student_image_size = student_image_size

        if isinstance(teacher_image_size, int):
            teacher_image_size = (teacher_image_size, teacher_image_size)
        if isinstance(student_image_size, int):
            student_image_size = (student_image_size, student_image_size)

        self.teacher_resize = iaa.Sequential([
            iaa.Resize({"height": teacher_image_size[0], "width": teacher_image_size[1]})
        ])
        self.student_resize = iaa.Sequential([
            iaa.Resize({"height": student_image_size[0], "width": student_image_size[1]})
        ])
        self.to_tensor = True
        self.compose = transforms.Compose([transforms.ToTensor()])

    def aug(self, image, points: list):
        bboxes = self.poly2bbox(points)
        shape = image.shape[:2]
        bboxes_ia = self.bbox_list2ia(shape, bboxes)
        image_aug, bbox_aug = self.seq(image=image, bounding_boxes=bboxes_ia)

        # If teacher model == CharNet
        h, w = shape
        scale = max(h, w) / float(self.teacher_image_size)
        image_resize_height = int(round(h / scale / 128) * 128)
        image_resize_width = int(round(w / scale / 128) * 128)
        scale_h = float(h) / image_resize_height
        scale_w = float(w) / image_resize_width
        scale_info = [scale_w, scale_h, w, h]

        self.teacher_resize = iaa.Sequential([
            iaa.Resize((image_resize_height, image_resize_width))
        ])
        self.teacher_resize = iaa.Sequential([
            iaa.Resize({"height": image_resize_height, "width": image_resize_width})
        ])

        teacher_image_aug, teacher_box_aug = self.teacher_resize(image=image_aug, bounding_boxes=bbox_aug)

        student_image_aug, student_box_aug = self.student_resize(image=image_aug, bounding_boxes=bbox_aug)

        if self.to_tensor:
            teacher_image_aug = self.img_ia2tensor(teacher_image_aug)
            teacher_box_aug = self.bbox_ia2tensor(teacher_box_aug)
            student_image_aug = self.img_ia2tensor(student_image_aug)
            student_box_aug = self.bbox_ia2tensor(student_box_aug)

        return teacher_image_aug, teacher_box_aug, student_image_aug, student_box_aug, scale_info

    @staticmethod
    def bbox_list2ia(shape, bbox: list):
        # convert Variable:bbox from List[List[], List[], ... ] to Imgaug BoundingBoxes.
        img_bbox = []
        for bbox_ in bbox:
            img_bbox.append(BoundingBox(x1=bbox_[0], y1=bbox_[1], x2=bbox_[2], y2=bbox_[3]))
        return BoundingBoxesOnImage(img_bbox, shape=shape)

    @staticmethod
    def poly2bbox(points_list):
        # convert points(polygon) to standard bounding box format
        bboxes = []
        for points in points_list:
            xs = [points[2*idx] for idx in range(4)]
            ys = [points[2*idx+1] for idx in range(4)]
            top_left_x = min(xs)
            top_left_y = min(ys)
            bottome_right_x = max(xs)
            bottome_right_y = max(ys)
            bboxes.append([top_left_x, top_left_y, bottome_right_x, bottome_right_y])
        return bboxes

    @staticmethod
    def bbox_ia2tensor(bbox_ia):
        temp = []
        for i in range(len(bbox_ia)):
            box_coords = bbox_ia.bounding_boxes[i]
            temp.append([box_coords.x1, box_coords.y1, box_coords.x2, box_coords.y2])
        return torch.tensor(temp).reshape(-1, 4)

    def img_ia2tensor(self, img_ia):
        img = self.compose(img_ia)
        return img


class ICDAR_Dataset(Dataset):
    def __init__(self, img_dir, annotation_dir, transform):
        """
        img_dir : path of directory containing images
        annotation_dir : path of directory containing annotation files (ICDAR2015 : .txt files)
        transform : Augment Class Instance
        """

        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.img_names = sorted(glob.glob(self.img_dir + '/*.jpg'), key=lambda x: int(x.split('img_')[-1].split('.')[0]))
        self.file_names = sorted(glob.glob(self.annotation_dir + '/*.txt'), key=lambda x: int(x.split('img_')[-1].split('.')[0]))
        assert len(self.img_names) == len(self.file_names), "# of images and # of annotations must be same."

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        with open(self.file_names[idx], 'r', encoding='utf-8-sig') as f:
            txt_lines = f.readlines()
            label = []
            for line in txt_lines:
                one_word = []
                vals = line.split(',')
                # print(type(vals[0]), vals[0])
                # if 'ff' in vals[0]:
                #     print('vals[0]', vals[0])
                #     vals[0] = vals[0][6:]
                for val in vals[:-1]:
                    one_word.append(int(float(val)))
                label.append(one_word)

        teacher_image_aug, teacher_label_aug, student_image_aug, student_label_aug, scale_info = self.transform.aug(image, label)
        return teacher_image_aug, teacher_label_aug, student_image_aug, student_label_aug, scale_info

    @staticmethod
    def collate_fn(batch):
        """
        Each bbox label has different shape (N, 4) (because each image has different # of objects)
        """
        teacher_images = []
        teacher_box_labels = []
        student_images = []
        student_box_labels = []
        charnet_scale_infos = []
        for b in batch:
            teacher_images.append(b[0])
            teacher_box_labels.append(b[1])
            student_images.append(b[2])
            student_box_labels.append(b[3])
            charnet_scale_infos.append(b[4])

        # teacher_images = torch.stack(teacher_images, dim=0)
        student_images = torch.stack(student_images, dim=0)
        # label -> torch.stack  : impossible
        return teacher_images, teacher_box_labels, student_images, student_box_labels, charnet_scale_infos


if __name__ == '__main__':
    """
    
    How to use "ICDAR_Dataset" & "Augment" ?
    EX)

    """
    pass
    # Check fordebug.py for reference
    # Old version
    # target_img_size = (640, 640)
    # custom_augment = Augment(img_size=target_img_size)
    # custom_dataset = ICDAR_Dataset(img_dir='ch4_training_images',
    #                                annotation_dir='ch4_training_localization_transcription_gt',
    #                                transform=custom_augment)
    #
    # one_img, one_label = custom_dataset[0]
    #
    # # Visualize Test
    # import numpy as np
    # from visualize import bbox_visualize
    # one_img = np.array(one_img) * 255
    # one_img = np.transpose(one_img, (1, 2, 0)).astype(np.uint8).copy()
    #
    # vis_img = bbox_visualize(one_img, one_label)
    # cv2.imshow('Example', vis_img)
    # cv2.waitKey(0)
    #
    # # Dataloader
    # from torch.utils.data import DataLoader
    #
    # custom_dataloader = DataLoader(custom_dataset, batch_size=4, shuffle=True, collate_fn=custom_dataset.collate_fn)
    #
    # for i, (img, label) in enumerate(custom_dataloader):
    #     print("This is index :", i)
    #     print("This is Img Tensor : ", type(img), img.shape)
    #     print("This is Label Tensor : ", len(label), label[0].shape, label[1].shape, label[2].shape, label[3].shape)
    #
        # break

    
    
        

