# 안되는 거
#python distill.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0
#python distill2_st2.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml
# distill_ln_4maskdiff

# 되는 거
# --device 0 / 1/ 2 ... or --device cpu
python distill2.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name strategy3
# distill_ln_4maskdiff_hint

python distill_xn_.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0

python distill_xn2_.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0

python distill_xn3_.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0


python distill_xn_maskdiff_PFI.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff_

# 이름 정리 후
# python distill.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0
python distill_ln_4maskdiff.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name ln_4maskdiff
python distill_ln_4maskdiff_Hint.py --cfg_teacher models/yolov5l.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/l/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name ln_4amskdiff_hint

python distill_xn_maskdiff.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff
python distill_xn_maskdiff_PFI.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff_PFI
python distill_xn_maskdiff_PFI_Hint.py --cfg_teacher models/yolov5x.yaml --cfg_student models/yolov5n.yaml --teacher_weights teacher/x/weights/best.pt --student_weights '' --data data/ICDAR.yaml --device 0 --name xn_maskdiff_PFI_Hint
