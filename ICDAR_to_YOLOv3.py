import os
import cv2
import glob
from tqdm import tqdm

img_dir = r'C:\Users\user\PycharmProjects\dino_lib22\ICDAR2015\ch4_test_images'
gt_dir = r'C:\Users\user\PycharmProjects\dino_lib22\ICDAR2015\Challenge4_Test_Task1_GT'
img_paths = sorted(glob.glob(img_dir + '/*.jpg'))
gt_paths = sorted(glob.glob(gt_dir + '/*.txt'))

for img_path, gt_path in tqdm(zip(img_paths, gt_paths)):
    img = cv2.imread(img_path)
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        gt_label = f.readlines()
    h, w = img.shape[:2]

    new_label = []
    for line in gt_label:
        one = '0 '
        vals = line.split(',')[:8]
        x_vals = [float(vals[2*i]) for i in range(4)]
        y_vals = [float(vals[2*i+1]) for i in range(4)]
        x_center = sum(x_vals) / (4 * w)
        y_center = sum(y_vals) / (4 * h)
        width = (max(x_vals) - min(x_vals)) / w
        height = (max(y_vals) - min(y_vals)) / h
        one += str(x_center)[:7] + ' '
        one += str(y_center)[:7] + ' '
        one += str(width)[:7] + ' '
        one += str(height)[:7] + ' '
        new_label.append(one + '\n')
    new_label[-1] = new_label[-1][:-1]

    new_label_path = os.path.join(r'C:\Users\user\PycharmProjects\dino_lib22\ICDAR2015\labels\test_icdar',
                                  os.path.basename(img_path).replace('.jpg', '.txt'))
    with open(new_label_path, 'w') as f:
        f.writelines(new_label)








