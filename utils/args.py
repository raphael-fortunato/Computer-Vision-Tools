import argparse


def classification_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model',
            type=str,
            required=False,
            help='Model name')
    parser.add_argument(
            '--path',
            type=str,
            required=False,
            help='path to the model to test')
    parser.add_argument(
            '--root',
            action='store_true',
            required=False,
            help='Database directory')
    parser.add_argument(
            '--seed',
            type=int,
            default=0,
            required=False,
            help='set random seed')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=24,
            required=False,
            help='Batch size')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=12,
            required=False,
            help='Number of dataloader workers')
    parser.add_argument(
            '--img_size',
            type=int,
            default=448,
            required=False,
            help='Image Size')
    parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            required=False,
            help='Image Size')
    parser.add_argument(
            '--lr',
            type=float,
            default=0.005,
            required=False,
            help='Learning Rate')
    parser.add_argument(
            '--lr_scheduler',
            type=str,
            default='plateau',
            required=False,
            help='Learning Rate scheduler')
    parser.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            required=False,
            help='Learning Rate')
    parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.0005,
            required=False,
            help='weight decay regularization')
    parser.add_argument(
            '--step_size',
            type=float,
            default=10,
            required=False,
            help='lr scheduler step size')
    parser.add_argument(
            '--gamma',
            type=float,
            default=.5,
            required=False,
            help='lr scheduler gamma')
    parser.add_argument(
            '--patience',
            type=int,
            default=3,
            required=False,
            help='number of epochs with no improvements before calling step()')

    return parser.parse_args()


def detection_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--path',
            type=str,
            required=False,
            help='path to the model to test')
    parser.add_argument(
            '--train_segmentation',
            action='store_true',
            required=False,
            help='use segmentation as data augmentation')
    parser.add_argument(
            '--train_bbx',
            action='store_true',
            required=False,
            help='use segmentation as data augmentation')
    parser.add_argument(
            '--seed',
            type=int,
            default=0,
            required=False,
            help='set random seed')
    parser.add_argument(
            '--epochs',
            type=int,
            default=30,
            required=False,
            help='Batch size')
    parser.add_argument(
            '--batch_size',
            type=int,
            default=4,
            required=False,
            help='Batch size')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
            required=False,
            help='Number of dataloader workers')
    parser.add_argument(
            '--lr',
            type=float,
            default=0.005,
            required=False,
            help='Learning Rate')
    parser.add_argument(
            '--momentum',
            type=float,
            default=0.9,
            required=False,
            help='Learning Rate')
    parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.0005,
            required=False,
            help='weight decay regularization')
    parser.add_argument(
            '--step_size',
            type=float,
            default=10,
            required=False,
            help='lr scheduler step size')
    parser.add_argument(
            '--gamma',
            type=float,
            default=.1,
            required=False,
            help='lr scheduler gamma')
    parser.add_argument(
            '--threshold',
            type=float,
            default=.5,
            required=False,
            help='threshold used for inference')
    return parser.parse_args()
