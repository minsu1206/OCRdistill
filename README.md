# OCRdistill
YAI (Yonsei University Artificial Intelligence) & LOMIN(https://lomin.ai/)


> Last Update: 22/04/14 by MINSU KIM
> <br>
> Team Member: <br>
> &nbsp; &nbsp; &nbsp; &nbsp; Minsu Kim (Leader) <br>
> &nbsp; &nbsp; &nbsp; &nbsp; Chanmi Lee <br> 
> &nbsp; &nbsp; &nbsp; &nbsp; Yongjun An <br>
> &nbsp; &nbsp; &nbsp; &nbsp; Sujung Park <br>


<br></br>
## Structure

~~~
ICDAR2015       # we just uploaded small size subset of ICDAR dataset.
    images
        train_icdar
            img_1.jpg
            ...
            img_9.jpg
            img_10.jpg
        test_icdar
            img_1.jpg
            ...
            img_9.jpg
            img_10.jpg
  labels
        train_icdar
            img_1.txt
            ...
            img_7.txt
            img_10.txt
        test_icdar
            img_1.txt
            ...
            img_7.txt
            img_10.txt
        test_icdar.cache
        test_icdar.cache.npy
        train_icdar.cache
        train_icdar.cache.npy
    
distill         # Heavily based on YOLOv5 Repo : https://github.com/ultralytics/yolov5
                # Here, only those that have been changed or modified are displayed.
    teacher
        n / s / m / l /x    # Teacher model trained on ICDAR 2015.
                            # Actually, YOLOv5 l and x can be teacher model, Otherwise are not appropriate.
                            # model weights can be downloaded by google drive links. (See below)
    utils
        loss.py
    distill.py
    distill_ln_4maskdiff.py
    distill_ln_4maskdiff_Hint.py
    distill_xn_maskdiff.py
    distill_xn_maskdiff_PFI.py
    distill_xn_maskdiff_PFI_Hint.py
    ...     # others are no used at our projects.
    
legacy  # not used

~~~
<br></br>
---

## Dataset
> ICDAR2015

<br></br>
---
<br></br>
## Project Introduction

#### Does Knowledge distillation for OCR (Text Detection) work well? Then, How?

> Our approach : Apply Knowledge distillation methods for object detection to Text Detection 

<br></br>
#### Text Detection Model
> YOLOv5 

<br></br>
#### Distilation Methods :
1. Hint learning
    - FitNets: Hints for thin deep nets .
2. Masking ROI features 
    - General Instance Distillation for Object Detection
    - Distilling Object Detectors via Decoupled Features
3. Prediction guided feature difference
    - Knowledge Distillation for Object Detection via Rank Mimicking and Prediction-guided Feature Imitation

<br></br>
#### Distillation Experiments :
File name with
- xn : Teacher model YOLOv5 X -> Student model YOLOv5 N
- ln : Teacher model YOLOv5 L -> Student model YOLOv5 N
- mask : Use Distillation Method [2]
- Hint : Use Distillation Method [1]
- PFI : Use Distillation Method [3]
<br></br>
---
<br></br>
## RUN CODE
> python distill_ln_4maskdiff.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name ln_4maskdiff

> python distill_ln_4maskdiff_Hint.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name ln_4amskdiff_hint

> python distill_xn_maskdiff.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff

> python distill_xn_maskdiff_PFI.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff_PFI

> python distill_xn_maskdiff_PFI_Hint.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff_PFI_Hint


---
<br></br>

## ICDAR2015 Pretrained Teacher Model
https://drive.google.com/drive/folders/1vmZ__-T78_myS-wIIAZscrraT-ixG5Y3?usp=sharing










