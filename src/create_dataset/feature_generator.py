#!/usr/bin/env python3
# coding: utf-8

import math

import numpy as np


def F2I(val, orig, scale):
    """Convert points in lidar coordinate system(axis aligned projection) to feature_map coordinate system."""
    return int(np.floor((orig - val) * scale))
    #return np.floor((orig - val) * scale).astype(np.uint8)

def GroupPc2Pixel(pc_x, pc_y, scale, range):
    """Convert points in lidar coordinate system(axis rotated projection) to feature_map coordinate system."""
    fx = (range - (0.707107 * (pc_x + pc_y))) * scale
    fy = (range - (0.707107 * (pc_x - pc_y))) * scale
    x = -1 if fx < 0 else int(fx)
    y = -1 if fy < 0 else int(fy)
    return x, y

def Pixel2pc(in_pixel, in_size, out_range):
    """Convert points in feature_map coordinate system to lidar coordinate system."""
    res = 2.0 * out_range / in_size
    return out_range - (in_pixel + 0.5) * res

class InputFeatureGenerator():
    def __init__(self, grid_range, width, height,
                 use_constant_feature, use_intensity_feature):
        self.range = grid_range
        self.width = int(width)
        self.height = int(height)
        self.siz = self.width * self.height
        self.min_height = -5.0
        self.max_height = 5.0
        self.use_constant_feature = use_constant_feature
        self.use_intensity_feature = use_intensity_feature

        self.log_table = np.zeros(256)
        for i in range(len(self.log_table)):
            self.log_table[i] = np.log1p(i)

        if self.use_constant_feature and self.use_intensity_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.direction_data = 3
            self.top_intensity_data = 4
            self.mean_intensity_data = 5
            self.distance_data = 6
            self.nonempty_data = 7
            self.feature = np.zeros((self.siz, 8), dtype=np.float16)

        elif self.use_constant_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.direction_data = 3
            self.distance_data = 4
            self.nonempty_data = 5
            self.feature = np.zeros((self.siz, 6), dtype=np.float16)

        elif self.use_intensity_feature:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.top_intensity_data = 3
            self.mean_intensity_data = 4
            self.nonempty_data = 5
            self.feature = np.zeros((self.siz, 6), dtype=np.float16)

        else:
            self.max_height_data = 0
            self.mean_height_data = 1
            self.count_data = 2
            self.nonempty_data = 3
            self.feature = np.zeros((self.siz, 4), dtype=np.float16)

        if self.use_constant_feature:
            for row in range(self.height):
                for col in range(self.width):
                    idx = row * self.width + col
                    center_x = Pixel2pc(row, self.height, self.range)
                    center_y = Pixel2pc(col, self.width, self.range)
                    self.feature[idx, self.direction_data] \
                        = np.arctan2(center_y, center_x) / (2.0 * np.pi)
                    self.feature[idx, self.distance_data] \
                        = np.hypot(center_x, center_y) / 60. - 0.5

    def logCount(self, count):
        if count < len(self.log_table):
            return self.log_table[count]
        else:
            return np.log(1 + count)

    def load_pc_from_file(self, pc_f):
        return np.fromfile(pc_f, dtype=np.float32, count=-1).reshape([-1, 4])

    def generate(self, points):
        self.map_idx = np.zeros(len(points))
        inv_res_x = 0.5 * self.width / self.range
        inv_res_y = 0.5 * self.height / self.range
        for i in range(len(points)):
            if points[i, 2] <= self.min_height or \
               points[i, 2] >= self.max_height:
                self.map_idx[i] = -1
            pos_x = F2I(points[i, 1], self.range, inv_res_x)
            pos_y = F2I(points[i, 0], self.range, inv_res_y)
            ## 2018.6.21, switch to axis rotated projection
            #pos_x, pos_y = GroupPc2Pixel(points[i, 1], points[i, 0], inv_res_x, self.range)
            if pos_x >= self.width or pos_x < 0 or \
               pos_y >= self.height or pos_y < 0:
                self.map_idx[i] = -1
                continue


            self.map_idx[i] = pos_y * self.width + pos_x
            idx = int(self.map_idx[i])
            pz = points[i, 2]
            pi = points[i, 3] / 255.0
            if self.feature[idx, self.max_height_data] < pz:
                self.feature[idx, self.max_height_data] = pz
                if self.use_intensity_feature:
                    self.feature[idx, self.top_intensity_data] = pi

            self.feature[idx, self.mean_height_data] += pz

            if self.use_intensity_feature:
                self.feature[idx, self.mean_intensity_data] += pi

            self.feature[idx, self.count_data] += 1.0

        for i in range(self.siz):
            eps = 1e-6
            if self.feature[i, self.count_data] < eps:
                self.feature[i, self.max_height_data] = 0.0
            else:
                self.feature[i, self.mean_height_data] \
                    /= self.feature[i, self.count_data]
                if self.use_intensity_feature:
                    self.feature[i, self.mean_intensity_data] \
                        /= self.feature[i, self.count_data]
                self.feature[i, self.nonempty_data] = 1.0
            self.feature[i, self.count_data] \
                = self.logCount(int(self.feature[i, self.count_data]))
                
        return self.feature


class OutputFeatureGenerator():
    def __init__(self, grid_range):
        self.grid_range = grid_range

    def points_in_box(self, corners, points, wlh_factor=1.0):
        """Check whether points are inside the box.

        Partially changed the function implemented in
        "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/geometry_utils.py"

        Picks one corner as reference (p1) and computes
        the vector to a target point (v).
        Then for each of the 3 axes, project v onto the axis
        and compare the length.
        Inspired by: https://math.stackexchange.com/a/1552579

        :param box: <Box>.
        :param points: <np.float: 3, n>.
        :param wlh_factor: Inflates or deflates the box.
        :return: <np.bool: n, >.

        """
        p1 = corners[:, 0]
        p_x = corners[:, 4]
        p_y = corners[:, 1]
        p_z = corners[:, 3]

        pi = p_x - p1
        pj = p_y - p1
        pk = p_z - p1

        v = points - np.array([[p1[0]], [p1[1]], [p1[2]]]).astype(np.float32)

        iv = np.dot(pi, v)
        jv = np.dot(pj, v)
        kv = np.dot(pk, v)

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(pi, pi))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(pj, pj))
        mask_z = np.logical_and(0 <= kv, kv <= np.dot(pk, pk))
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

        return mask

    def points_in_box2d(self, corners, box2d, points):
        """2D version of points_in_box"""
        p1 = box2d[0]
        p_x = box2d[1]
        p_y = box2d[3]

        pi = p_x - p1
        pj = p_y - p1

        v = points[:2] - p1

        iv = np.dot(pi, v)
        jv = np.dot(pj, v)

        mask_x = np.logical_and(0 <= iv, iv <= np.dot(pi, pi))
        mask_y = np.logical_and(0 <= jv, jv <= np.dot(pj, pj))
        mask = np.logical_and(mask_x, mask_y)

        return mask


    def generate(self, size, grid_centers, box_corners,
                    box2d, box2d_center, pc_points,
                    height_pt, label, yaw, out_feature):
        """Generate out_feature.

        Parameters
        ----------
        size : int
            feature map size
        grid_centers : numpy.ndarray
            center coordinates of feature_map grid
        box_corners : numpy.ndarray
            The coordinates of each corner of the object's box.
        box2d : numpy.ndarray
            The x,y coordinates of the object's box
        box2d_center : numpy.ndarray
            Center x,y coordinates of object's box
        pc_points : numpy.ndarray
            Input point cloud. (4, n)
        height_pt : float
            Height of object.
        label : int
            Object label. classify_pt
        yaw : float
            Rotation of the yaw of the object. heading_pt.
        out_feature : numpy.ndarray
            Output features. category, instance(x, y),
            confidence, classify, heading(x, y), height


        Returns
        -------
        out_feature : numpy.ndarray
            Output features. category, instance(x, y),
            confidence, classify, heading(x, y), height

        """

        box2d_left = box2d[:, 0].min()
        box2d_right = box2d[:, 0].max()
        box2d_top = box2d[:, 1].max()
        box2d_bottom = box2d[:, 1].min()

        inv_res = 0.5 * size / float(self.grid_range)
        res = 1.0 / inv_res
        max_length = abs(2 * res)

        search_area_left_idx = F2I(box2d_left, self.grid_range, inv_res)
        search_area_right_idx = F2I(box2d_right, self.grid_range, inv_res)
        search_area_top_idx = F2I(box2d_top, self.grid_range, inv_res)
        search_area_bottom_idx = F2I(box2d_bottom, self.grid_range, inv_res)
        # search_area_left_idx = F2I(box2d_bottom, box2d_left, self.grid_range, inv_res)
        # search_area_right_idx = F2I(box2d_top, box2d_right, self.grid_range, inv_res)
        # search_area_top_idx = F2I(box2d_top, self.grid_range, inv_res)
        # search_area_bottom_idx = F2I(box2d_bottom, self.grid_range, inv_res)



        num_points = np.count_nonzero(self.points_in_box(box_corners, pc_points[:3, :]))
        if num_points < 4 and label == 0:
            return out_feature
        elif num_points < 4 and label == 1:
            return out_feature
        elif num_points < 4 and label == 2:
            return out_feature
        elif num_points < 4 and label == 3:
            return out_feature
        elif num_points < 4 and label == 4:
            return out_feature

        for i in range(search_area_right_idx - 1, search_area_left_idx + 1):
            for j in range(search_area_top_idx - 1, search_area_bottom_idx + 1):
                if 0 <= i and i < size and 0 <= j and j < size:
                    grid_center_x = Pixel2pc(i, float(size), self.grid_range)
                    grid_center_y = Pixel2pc(j, float(size), self.grid_range)

                    if max_length < np.abs(box2d_center[0] - grid_center_x):
                        x_scale = max_length / \
                            np.abs(box2d_center[0] - grid_center_x)
                    else:
                        x_scale = 1.
                    if max_length < np.abs(box2d_center[1] - grid_center_y):
                        y_scale = max_length / \
                            np.abs(box2d_center[1] - grid_center_y)
                    else:
                        y_scale = 1.

                    normalized_yaw = math.atan(math.sin(yaw) / math.cos(yaw))

                    mask = self.points_in_box2d(
                        box_corners, box2d,
                        np.array([grid_center_x, grid_center_y, 0]).astype(np.float32))

                    if mask:
                        out_feature[i, j, 0] = 1.                               # category_pt
                        out_feature[i, j, 1] = ((box2d_center[0] - grid_center_x) * -1) * min(x_scale, y_scale) #instance_pt_x
                        out_feature[i, j, 2] = ((box2d_center[1] - grid_center_y) * -1) * min(x_scale, y_scale) #instance_pt_y
                        out_feature[i, j, 3] = 1.                               # confidence_pt
                        out_feature[i, j, 4] = label                            # classify_pt
                        out_feature[i, j, 5] = math.cos(normalized_yaw * 2.0)   #heading_pt_x
                        out_feature[i, j, 6] = math.sin(normalized_yaw * 2.0)   #heading_pt_y
                        out_feature[i, j, 7] = height_pt                        # height_pt

        return out_feature