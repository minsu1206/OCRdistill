"""
model build

    Teacher model
        1. CharNet
        2.

    Student model
        1. YOLOv3
        2. SSDLite

"""


class ModelBuild:
    def __init__(self, args):
        self.args = args

    def teacher_model(self):
        name = self.args.T_model

        # TODO
        if name == 'TextFuseNet_resnext101':

            raise NotImplementedError()
            # return

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





