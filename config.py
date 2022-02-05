import argparse


def get_parse():
    """
    argument (인자)를 넣어줄 때 보통

        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_path', type='str', ...)
        ...
        args = parser.parse_args()

    와 같은 방식으로 작성하는 경우가 많은데, 설정해야 할 값들이 많을 경우 명령창이 길어져서 조금 불편합니다.
    이렇게 Namespace 로 따로 빼놓으면 configuration 을 dictionary 처럼 미리 작성해놓을 수 있어서 편합니다.

    """
    args = argparse.Namespace()

    # Train Dataset config
    args.dataset_path = ''
    args.annotation_path = ''
    args.split_ratio = 0.8
    args.img_size = 1280

    # Training config - KD
    args.T_model = 'TextFuseNet_resnext101'
    args.S_model = 'CRAFT'
    args.method = [0]

    # Training config - settings
    args.use_gpu = True
    args.batch_size = 8
    args.epoch = 200
    args.learning_rate = 0.001


    return args


if __name__ == "__main__":

    args = get_parse()
    print(args.T_model)
    print(args.S_model)
    print(args.dataset_path)



