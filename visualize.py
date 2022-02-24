import cv2
import numpy as np

def bbox_visualize(img, bbox):
    """
    img : image ndarray
    bbox : List[List(top-left-x, top-left-y, bottom-right-x, bottom-right-y), List(), ...] or
            Torch.Tensor([List(top-left-x, top-left-y, bottom-right-x, bottom-right-y), List(), ...])

    Return
        Visualized bbox
    """
    # img = np.uint8(img).copy()
    print(type(img), img.shape)
    color = (255, 0, 0)
    thickness = 2
    for one_box in bbox:
        start_point = (int(one_box[0]), int(one_box[1]))
        end_point = (int(one_box[2]), int(one_box[3]))
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
    
    return img



    