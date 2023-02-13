import argparse
import copy
import os
import warnings

import mmcv
import torch
from torchpack.utils.config import configs
from torchpack import distributed as dist
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.utils import recursive_eval



def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase"
        "the inference speed",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="Format the output results without perform evaluation. It is"
        "useful when you want to format the result to a specific format and "
        "submit it to the test server",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show-dir", help="directory where results will be saved")
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results.",
    )
    parser.add_argument(
        "--tmpdir",
        help="tmp directory used for collecting results from multiple "
        "workers, available when gpu-collect is not specified",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function (deprecate), "
        "change to --eval-options instead.",
    )
    parser.add_argument(
        "--eval-options",
        nargs="+",
        action=DictAction,
        help="custom options for evaluation, the key-value pair in xxx=yyy "
        "format will be kwargs for dataset.evaluate() function",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            "--options and --eval-options cannot be both specified, "
            "--options is deprecated in favor of --eval-options"
        )
    if args.options:
        warnings.warn("--options is deprecated in favor of --eval-options")
        args.eval_options = args.options
    return args


def main():
    args = parse_args()
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    assert args.out or args.eval or args.format_only or args.show or args.show_dir, (
        "Please specify at least one operation (save/eval/format/show the "
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"'
    )

    if args.eval and args.format_only:
        raise ValueError("--eval and --format_only cannot be both specified")

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    print(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test]
        )
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    distributed = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build data
    get_data()

def get_sensor_calib_data(cam):
    sensor_data = {
        'CAM_BACK': {
            "translation": [
                -2.0,
                -0.0,
                2.8
            ],
            "rotation": [
                0.5,
                -0.5,
                -0.5,
                0.5
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT': {
            "translation": [
                1.5,
                -0.0,
                2.8
            ],
            "rotation": [
                0.5,
                -0.5,
                0.5,
                -0.5
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT_LEFT': {
            "translation": [
                1.3,
                0.4,
                2.8
            ],
            "rotation": [
                0.6743797,
                -0.6743797,
                0.2126311,
                -0.2126311
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_FRONT_RIGHT': {
            "translation": [
                1.3,
                -0.4,
                2.8
            ],
            "rotation": [
                0.2126311,
                -0.2126311,
                0.6743797,
                -0.6743797
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_BACK_LEFT': {
            "translation": [
                -0.85,
                0.4,
                2.8
            ],
            "rotation": [
                0.6963642,
                -0.6963642,
                -0.1227878,
                0.1227878
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'CAM_BACK_RIGHT': {
            "translation": [
                -0.85,
                -0.4,
                2.8
            ],
            "rotation": [
                -0.1227878,
                0.1227878,
                0.6963642,
                -0.6963642
            ],
            "camera_intrinsic": [
                [
                    1142.5184053936916,
                    0.0,
                    800.0
                ],
                [
                    0.0,
                    1142.5184053936916,
                    450.0
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        },
        'LIDAR_TOP': {
            "translation": [
                0.94,
                -0.0,
                2.8
            ],
            "rotation": [
                1.0,
                0.0,
                0.0,
                0.0
            ],
            "camera_intrinsic": []
        }
    }

    ego_pose = {
        "translation": [
            88.38433074951172,
            -88.25868225097656,
            0.0
        ],
        "rotation": [
            0.7076030964832982,
            6.146967038796277e-06,
            5.149090818303537e-06,
            -0.7066101172378937
        ],
    }

    return sensor_data, ego_pose

def get_sample_data():
    vehicle = 'model3'
    date = '01-08-2023'
    timestamp = '1673120610692'
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]
    # read images
    images = {}
    for cam in camera_types:
        img_path = f'{vehicle}-{date}_{cam}_{timestamp}.jpg'
        images[cam] = img_path

    # read lidar points
    points_file = f'{vehicle}-{date}_LIDAR_TOP_{timestamp}.pcd'

    # read cam intrinsics
    sensor_calib_data, ego_pose  = get_sensor_calib_data()
    all_cam_intrinsics = {}
    cam_translations = {}
    cam_rotations = {}
    for cam in camera_types:
        all_cam_intrinsics[cam] = sensor_calib_data[cam]['camera_intrinsic']
        cam_translations[cam] = sensor_calib_data[cam]['translation']
        cam_rotations[cam] = sensor_calib_data[cam]['rotations']

    return images, points_file, all_cam_intrinsics, sensor_calib_data['LIDAR_TOP']['translation'], sensor_calib_data['LIDAR_TOP']['rotation'], \
        ego_pose['translation'], ego_pose['rotation'], cam_translations, cam_rotations


def get_data():
    images, points, all_cam_intrinsics, lidar2ego_translation, lidar2ego_rotation,
    ego2global_translation, ego2global_rotation, cam_translations, cam_rotations = get_sample_data()
    info = {
        "points": points,
        "cams": dict(),
        "lidar2ego_translation": lidar2ego_translation,
        "lidar2ego_rotation": lidar2ego_rotation,
        "ego2global_translation": ego2global_translation,
        "ego2global_rotation": ego2global_rotation,
    }


    l2e_r = info["lidar2ego_rotation"]
    l2e_t = info["lidar2ego_translation"]
    e2g_r = info["ego2global_rotation"]
    e2g_t = info["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    # obtain 6 image's information per frame
    camera_types = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    for cam in camera_types:
        cam_intrinsics = all_cam_intrinsics[cam]
        cam_info = obtain_sensor2top(
            l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam, cam_translations, cam_rotations,
        )
        cam_info.update(camera_intrinsics=camera_intrinsics)
        info["cams"].update({cam: cam_info})
    
    # obtain sweeps
    sweep = obtain_sensor2top(
        l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
    )
    info["sweeps"] = [sweep]


def obtain_sensor2top(
    l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar", cam_translations=None, cam_rotations=None
):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sweep = {
        "data": data
        "type": sensor_type,
        "sensor2ego_translation": cam_translations[sensor_type],
        "sensor2ego_rotation": cam_rotations[sensor_type],
        "ego2global_translation": e2g_t,
        "ego2global_rotation": e2g_r,
    }
    l2e_r_s = sweep["sensor2ego_rotation"]
    l2e_t_s = sweep["sensor2ego_translation"]
    e2g_r_s = sweep["ego2global_rotation"]
    e2g_t_s = sweep["ego2global_translation"]

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
    )
    T -= (
        e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
        + l2e_t @ np.linalg.inv(l2e_r_mat).T
    )
    sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
    sweep["sensor2lidar_translation"] = T
    return sweep