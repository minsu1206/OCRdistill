"""
    YOLOv3 학습을 위해선 COCO format 이 필요
    ICDAR 에서 제공하는 GT로 학습시키려면 새로운 dataload를 구성해야 하는데 그보단
    주어진 txt format 파일들을 COCO format으로 바꾸는 것이 더 편함

    category: 0 만 사용 (모두 Text)

"""

import json
import os
import glob
import cv2
from tqdm import tqdm
import copy
import random
random.seed(2022)

coco_default_format = {
    "info": {},     # NO need
    "images": [],
    "annotations": [],
    "licenses": [{"id": 0, "name": 'minsu_ICDAR', "url": ''}],  # NO need
    "categories": [{"id": 0, "name": 'Text', "supercategory": "Text"}]
}

img_folder_path = '../ICDAR2015/ch4_training_images'    # for TRAIN dataset
# img_folder_path = '../ICDAR2015/ch4_test_images'  # for TEST dataset
img_files_path = sorted(glob.glob(img_folder_path + '/*.jpg'))
gt_folder_path = '../ICDAR2015/ch4_training_localization_transcription_gt'  # for TRAIN dataset
# gt_folder_path = '../ICDAR2015/Challenge4_Test_Task1_GT'    # for TEST dataset
gt_files_path = sorted(glob.glob(gt_folder_path + '/*.txt'))

assert len(img_files_path) == len(gt_files_path)

label_count = 0
for img_file_path, gt_file_path in tqdm(zip(img_files_path, gt_files_path)):

    with open(gt_file_path, 'r', encoding='utf-8-sig') as f:        # utf-8-sig 아니면 오류나요!
        gt_label = f.readlines()
    # print(gt_label)

    img = cv2.imread(img_file_path)

    img_name = os.path.basename(img_file_path)
    img_id = int(img_name.split('.')[0].split('_')[-1])
    h, w = img.shape[:2]
    img_json = {"id": img_id, "width": w, "height": h, "file_name": img_name,
                "license": 0, "flickr_url": '', "coco_url": '', "date_captured": ''}
    coco_default_format["images"].append(img_json)
    detections = []
    for gt_label_line in gt_label:
        text_polygon = gt_label_line.split(',')[:8]
        segmentation = [int(text_polygon[i]) for i in range(8)]
        seg_x = [segmentation[i*2] for i in range(4)]
        seg_y = [segmentation[i*2 + 1]for i in range(4)]
        bbox = [min(seg_x), min(seg_y), max(seg_x), max(seg_y)]
        area = abs(bbox[0] - bbox[2]) * abs(bbox[1] - bbox[3])
        one_box_label = {
            "image_id": img_id,
            "id": label_count,
            "category_id": 0,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        }
        coco_default_format["annotations"].append(one_box_label)

        label_count += 1

with open('../ICDAR2015/COCOver/Train/annotations/ICDAR2015_Train_verCOCO.json', 'w') as f:
    json.dump(coco_default_format, f, indent=4, sort_keys=True)




