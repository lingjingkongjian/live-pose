# Still WIP
# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import os.path as osp
import tarfile
from argparse import ArgumentParser
from glob import glob
from urllib.request import urlretrieve

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter

from mmpose.apis.inference import init_pose_model
from mmpose.core.post_processing import get_affine_transform
from mmpose.datasets.dataset_info import DatasetInfo
from mmpose.datasets.pipelines import Compose

model = None
pipeline = None

def setup(img1):
    global model
    global pipeline

    model = init_pose_model(
        config_dict, args.pose_model_checkpoint, device=args.device.lower())
    pipeline = [
        dict(
            type='MultiItemProcess',
            pipeline=[
                dict(type='ToTensor'),
                dict(
                    type='NormalizeTensor',
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ]),
        dict(type='DiscardDuplicatedItems', keys_list=['sample_id']),
        dict(
            type='Collect',
            keys=['img'],
            meta_keys=['sample_id', 'camera', 'center', 'scale',
                       'image_file']),
    ]
    pipeline = Compose(pipeline)


def inference(imgs, cgroup):
    img1 = imgs[0]
    img2 = imgs[1]
    multiview_data = {}
    image_infos = []
    for c in range(2):
        img = imgs[c]
        height, width, _ = img.shape
        input_size = config_dict['model']['human_detector']['image_size']
        center = np.array((width / 2, height / 2), dtype=np.float32)
        scale = get_scale(input_size, (width, height))
        mat_input = get_affine_transform(
            center=center,
            scale=scale / 200.0,
            rot=0.0,
            output_size=input_size)
        img = cv2.warpAffine(img, mat_input,
                                (int(input_size[0]), int(input_size[1])))
        image_infos.append(input_data[i * num_cameras + c])

        singleview_data['img'] = img
        singleview_data['center'] = center
        singleview_data['scale'] = scale
        multiview_data[c] = singleview_data

    multiview_data = pipeline(multiview_data)
    # TODO: inference with input_heatmaps/kpts_2d
    multiview_data = collate([multiview_data], samples_per_gpu=1)
    multiview_data = scatter(multiview_data, [args.device])[0]
    with torch.no_grad():
        model.show_result(
            **multiview_data,
            input_heatmaps=None,
            dataset_info=dataset_info,
            radius=args.radius,
            thickness=args.thickness,
            out_dir=args.out_img_root,
            show=args.show,
            visualize_2d=args.visualize_single_view)

    return points_flat, p3ds_flat


def get_scale(target_size, raw_image_size):
    w, h = raw_image_size
    w_resized, h_resized = target_size
    if w / w_resized < h / h_resized:
        w_pad = h / h_resized * w_resized
        h_pad = h
    else:
        w_pad = w
        h_pad = w / w_resized * h_resized

    scale = np.array([w_pad, h_pad], dtype=np.float32)

    return scale
