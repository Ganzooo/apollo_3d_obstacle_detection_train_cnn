#!/usr/bin/env python3
# coding: utf-8

import argparse
import copy
import math
import os
import sys

import numba
import numpy as np
from pyquaternion import Quaternion
import matplotlib
import matplotlib.pyplot as pp
from feature_generator import OutputFeatureGenerator, InputFeatureGenerator
#import feature_generator_pb as fgpb


try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
except ImportError:
    for path in sys.path:
        if '/opt/ros/' in path:
            print('sys.path.remove({})'.format(path))
            sys.path.remove(path)
            from nuscenes.nuscenes import NuScenes
            from nuscenes.utils.data_classes import LidarPointCloud
            sys.path.append(path)
            break

def add_noise_points(points, num_rand_samples=5,
                     min_distance=5, sigma=2, add_noise_rate=0.1):
    """Add noise to the point cloud

    Parameters
    ----------
    points : numpy.ndarray
        Input point cloud. (n, 4)
    num_rand_samples : int, optional
        How many sample points to take at one angle, by default 5
    min_distance : int, optional
        Closest distance of noise, by default 5
    sigma : int, optional
        normal distribution of z, by default 2
    add_noise_rate : float, optional
        Percentage of angles to add noise, by default 0.1

    Returns
    -------
    points : numpy.ndarray
        Point cloud with added noise. (n, 4)

    """
    max_height = np.max(points[:, 2])
    min_height = np.min(points[:, 2])
    mean_height = np.mean(points[:, 2])

    distances = np.linalg.norm(
        np.hstack([points[:, 0:1], points[:, 1:2]]), axis=1)
    max_distance = np.max(distances)

    noise_points = []
    for theta in range(360):
        if np.random.rand() > add_noise_rate:
            continue
        distance = np.min(np.random.uniform(
            min_distance, max_distance, num_rand_samples))
        z = np.random.normal(mean_height, sigma)
        while min_height > z or z > max_height:
            z = np.random.normal(mean_height, sigma)
        x = distance * np.cos(theta)
        y = distance * np.sin(theta)
        i = points[np.random.randint(0, points.shape[0]), 3]
        noise_points.append([x, y, z, i])

    noise_points = np.array(noise_points)
    points = np.vstack([points, noise_points])

    return points


def create_dataset(dataroot, save_dir, width=864, height=864, grid_range=90.,
                   nusc_version='v1.0-mini',
                   use_constant_feature=False, use_intensity_feature=False,
                   end_id=None, augmentation_num=0, add_noise=False):
    """Create a learning dataset from Nuscens

    Parameters
    ----------
    dataroot : str
        Nuscenes dataroot path.
    save_dir : str
        Dataset save directory.
    width : int, optional
        feature map width, by default 864
    height : int, optional
        feature map height, by default 864
    grid_range : float, optional
        feature map range, by default 90.
    nusc_version : str, optional
        Nuscenes version. v1.0-mini or v1.0-trainval, by default 'v1.0-mini'
    use_constant_feature : bool, optional
        Whether to use constant feature, by default False
    use_intensity_feature : bool, optional
        Whether to use intensity feature, by default True
    end_id : int, optional
        How many data to generate. If None, all data, by default None
    augmentation_num : int, optional
        How many data augmentations for one sample, by default 0
    add_noise : bool, optional
        Whether to add noise to pointcloud, by default True

    Raises
    ------
    Exception
        Width and height are not equal

    """
    os.makedirs(os.path.join(save_dir, 'in_feature'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'out_feature'), exist_ok=True)

    nusc = NuScenes(
        version=nusc_version,
        dataroot=dataroot, verbose=True)
    ref_chan = 'LIDAR_TOP'

    if width == height:
        size = width
    else:
        raise Exception(
            'Currently only supported if width and height are equal')

    grid_length = 2. * grid_range / size
    z_trans_range = 0.5
    sample_id = 0
    data_id = 0
    grid_ticks = np.arange(
        -grid_range, grid_range + grid_length, grid_length)
    grid_centers \
        = (grid_ticks + grid_length / 2)[:len(grid_ticks) - 1]

    for my_scene in nusc.scene:
        first_sample_token = my_scene['first_sample_token']
        token = first_sample_token

        # try:
        while(token != ''):
            print('sample:{} {} created_data={}'.format(
                sample_id, token, data_id))
            my_sample = nusc.get('sample', token)
            sd_record = nusc.get(
                'sample_data', my_sample['data'][ref_chan])
            sample_rec = nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            pc_raw, _ = LidarPointCloud.from_file_multisweep(
                nusc, sample_rec, chan, ref_chan, nsweeps=1)

            _, boxes_raw, _ = nusc.get_sample_data(
                sd_record['token'], box_vis_level=0)

            z_trans = 0
            q = Quaternion()
            for augmentation_idx in range(augmentation_num + 1):
                pc = copy.copy(pc_raw)
                if add_noise:
                    pc.points = add_noise_points(pc.points.T).T
                boxes = copy.copy(boxes_raw)
                if augmentation_idx > 0:
                    z_trans = (np.random.rand() - 0.5) * 2 * z_trans_range
                    pc.translate([0, 0, z_trans])

                    z_rad = np.random.rand() * np.pi * 2
                    q = Quaternion(axis=[0, 0, 1], radians=z_rad)
                    pc.rotate(q.rotation_matrix)

                pc_points = pc.points.astype(np.float32)

                out_feature = np.zeros((size, size, 8), dtype=np.float32)
                for box_idx, box in enumerate(boxes):
                    if augmentation_idx > 0:
                        box.translate([0, 0, z_trans])
                        box.rotate(q)

                    label = 0
                    if box.name.split('.')[0] == 'vehicle':
                        if box.name.split('.')[1] == 'car':
                            label = 1
                        elif box.name.split('.')[1] == 'bus':
                            label = 2
                        elif box.name.split('.')[1] == 'truck':
                            label = 2
                        elif box.name.split('.')[1] == 'construction':
                            label = 2
                        elif box.name.split('.')[1] == 'emergency':
                            label = 2
                        elif box.name.split('.')[1] == 'trailer':
                            label = 2
                        elif box.name.split('.')[1] == 'bicycle':
                            label = 3
                        elif box.name.split('.')[1] == 'motorcycle':
                            label = 3
                    elif box.name.split('.')[0] == 'human':
                        label = 4
                    # elif box.name.split('.')[0] == 'movable_object':
                    #     label = 1
                    # elif box.name.split('.')[0] == 'static_object':
                    #     label = 1
                    else:
                        continue
                    height_pt = np.linalg.norm(
                        box.corners().T[0] - box.corners().T[3])
                    box_corners = box.corners().astype(np.float32)
                    corners2d = box_corners[:2, :]
                    box2d = corners2d.T[[2, 3, 7, 6]]
                    box2d_center = box2d.mean(axis=0)
                    yaw, pitch, roll = box.orientation.yaw_pitch_roll
                    out_feature_generator = OutputFeatureGenerator(grid_range)
                    out_feature = out_feature_generator.generate(
                        size=size, grid_centers=grid_centers, box_corners=box_corners,
                        box2d=box2d, box2d_center=box2d_center, pc_points=pc_points,
                        height_pt=height_pt, label=label, yaw=yaw, out_feature=out_feature)

                if use_constant_feature and use_intensity_feature:
                    channels = 8
                elif use_constant_feature or use_intensity_feature:
                    channels = 6
                else:
                    channels = 4
                
                ###pybind 11 USE:
                #feature_generator = fgpb.FeatureGenerator(
                #    grid_range, size, size)
                #in_feature = feature_generator.generate(
                #    pc_points.T,
                #    use_constant_feature, use_intensity_feature)
                #in_feature = np.array(in_feature).reshape(
                #     channels, size, size).astype(np.float16)

                in_feature_generator = InputFeatureGenerator(
                     grid_range, width, height,
                     use_constant_feature, use_intensity_feature)
                in_feature = in_feature_generator.generate(
                    pc_points.T)     
                in_feature = np.array(in_feature).reshape(
                    size, size, channels)

                in_feature = in_feature.transpose(0, 1, 2)

                np.save(os.path.join(
                    save_dir, 'in_feature/{:05}'.format(data_id)), in_feature)
                np.save(os.path.join(
                    save_dir, 'out_feature/{:05}'.format(data_id)), out_feature)
                token = my_sample['next']
                data_id += 1

                if data_id == end_id:
                    return
            sample_id += 1
        # except KeyboardInterrupt:
        #     return
        # except BaseException:
        #     print('skipped')
        #     continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataroot', '-dr', type=str,
                        help='Nuscenes dataroot path',
                        #default='/dataset/nuScenes/v1.0-trainval')
                        default='/dataset/nuScenes/v1.0-mini')
    parser.add_argument('--save_dir', '-sd', type=str,
                        help='Dataset save directory',
                        #default='/dataset/nuScenes/FeatureExtracted/v1.0-trainval')
                        default='/dataset_sub/nuScenes/FeatureExtracted2/v1.0-mini')
                        #default='/dataset_sub/nuScenes/FeatureExtracted/v1.0-trainval')
    parser.add_argument('--lidar_channel', type=int,
                        help='Lidar channel number: 32, 64, 128',
                        default=128)                        
    parser.add_argument('--width', type=int,
                        help='feature map width',
                        default=864)
    parser.add_argument('--height', type=int,
                        help='feature map height',
                        default=864)
    parser.add_argument('--range', type=int,
                        help='feature map range',
                        default=90)
    parser.add_argument('--nusc_version', type=str,
                        help='Nuscenes version. v1.0-mini or v1.0-trainval',
                        #default='v1.0-trainval')
                        default='v1.0-mini')
    parser.add_argument('--use_constant_feature', type=int,
                        help='Whether to use constant feature',
                        default=0)
    parser.add_argument('--use_intensity_feature', type=int,
                        help='Whether to use intensity feature',
                        default=0)
    parser.add_argument('--end_id', type=int,
                        help='How many data to generate. If None, all data',
                        default=1)
    parser.add_argument('--augmentation_num', '-an', type=int,
                        help='How many data augmentations for one sample',
                        default=0)
    parser.add_argument('--add_noise', type=int,
                        help='Whether to add noise to pointcloud',
                        default=0)

    args = parser.parse_args()
    
    if args.lidar_channel == 128:
        args.width = 864
        args.height = 864
        args.range = 90
        args.use_constant_feature = 0
        args.use_intensity_feature = 0
    else:
        args.width = 672
        args.height = 672
        args.range = 70
        args.use_constant_feature = 0
        args.use_intensity_feature = 1

    create_dataset(dataroot=args.dataroot,
                   save_dir=args.save_dir,
                   width=args.width,
                   height=args.height,
                   grid_range=args.range,
                   nusc_version=args.nusc_version,
                   use_constant_feature=args.use_constant_feature,
                   use_intensity_feature=args.use_intensity_feature,
                   end_id=args.end_id,
                   augmentation_num=args.augmentation_num,
                   add_noise=args.add_noise)