# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import open3d as o3d

from kiss_icp.scan import LidarScan, coerce_scan
from kiss_slam.kiss_slam_pybind import kiss_slam_pybind


class VoxelMap:
    def __init__(self, voxel_size: float):
        self.map = kiss_slam_pybind._VoxelMap(voxel_size)

    def integrate_frame(self, points, pose: np.ndarray, intensities=None):
        scan = coerce_scan(points, timestamps=np.array([]), intensities=intensities)
        vector3fvector = kiss_slam_pybind._Vector3fVector(scan.points.astype(np.float32))
        if scan.has_intensity:
            self.map._integrate_frame(vector3fvector, scan.intensities.astype(np.float32), pose)
        else:
            self.map._integrate_frame(vector3fvector, pose)

    def add_points(self, points, intensities=None):
        scan = coerce_scan(points, timestamps=np.array([]), intensities=intensities)
        vector3fvector = kiss_slam_pybind._Vector3fVector(scan.points.astype(np.float32))
        if scan.has_intensity:
            self.map._add_points(vector3fvector, scan.intensities.astype(np.float32))
        else:
            self.map._add_points(vector3fvector)

    def point_cloud(self):
        return np.asarray(self.map._point_cloud()).astype(np.float64)

    def point_cloud_with_intensity(self):
        points, intensities = self.map._point_cloud_with_intensity()
        intensities = np.asarray(intensities, dtype=np.float32)
        return LidarScan(
            points=np.asarray(points, dtype=np.float64),
            timestamps=np.array([]),
            intensities=intensities if intensities.size else None,
        )

    def clear(self):
        self.map._clear()

    def num_voxels(self):
        return self.map._num_voxels()

    def open3d_pcd_with_normals(self):
        points, normals, intensities = self.map._per_voxel_point_and_normal()
        # Reduce memory footprint
        pcd = o3d.t.geometry.PointCloud()
        pcd.point.positions = o3d.core.Tensor(np.asarray(points), o3d.core.Dtype.Float32)
        pcd.point.normals = o3d.core.Tensor(np.asarray(normals), o3d.core.Dtype.Float32)
        intensity_array = np.asarray(intensities, dtype=np.float32)
        if intensity_array.size and np.all(np.isfinite(intensity_array)):
            pcd.point.intensity = o3d.core.Tensor(
                intensity_array.reshape(-1, 1), o3d.core.Dtype.Float32
            )
        return pcd
