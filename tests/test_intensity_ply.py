import numpy as np
import open3d as o3d
import pytest

from kiss_icp.scan import LidarScan
from kiss_slam.config import OccupancyMapperConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.voxel_map import VoxelMap


def test_local_map_ply_contains_intensity_attribute(tmp_path):
    voxel_map = VoxelMap(1.0)
    voxel_map.add_points(
        np.array(
            [
                [0.1, 0.1, 0.1],
                [0.4, 0.1, 0.1],
                [0.1, 0.4, 0.1],
            ]
        ),
        intensities=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )

    graph = LocalMapGraph()
    graph.last_local_map.local_trajectory.append(np.eye(4))
    graph.finalize_local_map(voxel_map)

    filename = tmp_path / "local_map_with_intensity.ply"
    graph[0].write(str(filename))

    loaded = o3d.t.io.read_point_cloud(str(filename))
    loaded_intensity = loaded.point["intensity"].numpy().ravel()
    assert loaded_intensity.shape == (1,)
    assert np.allclose(loaded_intensity, [2.0])


def test_occupancy_ply_contains_intensity_attribute(tmp_path):
    mapper = OccupancyGridMapper(OccupancyMapperConfig(resolution=1.0, occupied_threshold=0.6))

    mapper.integrate_frame(
        LidarScan(
            points=np.array(
                [
                    [0.1, 0.1, 0.1],
                    [1.1, 0.1, 0.1],
                    [0.1, 1.1, 0.1],
                ]
            ),
            timestamps=np.array([]),
            intensities=np.array([1.0, 4.0, 7.0], dtype=np.float32),
        ),
        np.eye(4),
    )
    mapper.integrate_frame(
        LidarScan(
            points=np.array(
                [
                    [0.1, 0.1, 0.1],
                    [1.1, 0.1, 0.1],
                    [0.1, 1.1, 0.1],
                ]
            ),
            timestamps=np.array([]),
            intensities=np.array([3.0, 6.0, 9.0], dtype=np.float32),
        ),
        np.eye(4),
    )

    mapper.compute_3d_occupancy_information()
    mapper.write_3d_occupancy_grid(str(tmp_path))

    loaded = o3d.t.io.read_point_cloud(str(tmp_path / "occupancy_pcd.ply"))
    loaded_intensity = np.sort(loaded.point["intensity"].numpy().ravel())
    assert loaded_intensity.shape == (3,)
    assert np.allclose(loaded_intensity, [2.0, 5.0, 8.0])
    loaded_normals = loaded.point["normals"].numpy()
    assert loaded_normals.shape == (3, 3)
    assert (tmp_path / "occupancy_grid_bonxai.bin").exists()


def test_occupancy_ply_without_intensity_omits_attribute(tmp_path):
    mapper = OccupancyGridMapper(OccupancyMapperConfig(resolution=1.0, occupied_threshold=0.6))
    mapper.integrate_frame(
        np.array(
            [
                [0.1, 0.1, 0.1],
                [1.1, 0.1, 0.1],
                [0.1, 1.1, 0.1],
            ]
        ),
        np.eye(4),
    )

    mapper.compute_3d_occupancy_information()
    mapper.write_3d_occupancy_grid(str(tmp_path))

    loaded = o3d.t.io.read_point_cloud(str(tmp_path / "occupancy_pcd.ply"))
    with pytest.raises(KeyError):
        loaded.point["intensity"]


def test_occupancy_ply_can_disable_normals_and_bonxai_volume(tmp_path):
    mapper = OccupancyGridMapper(
        OccupancyMapperConfig(
            resolution=1.0,
            occupied_threshold=0.6,
            export_bonxai_volume=False,
            export_normals=False,
        )
    )
    mapper.integrate_frame(
        np.array(
            [
                [0.1, 0.1, 0.1],
                [1.1, 0.1, 0.1],
                [0.1, 1.1, 0.1],
                [1.1, 1.1, 0.1],
            ]
        ),
        np.eye(4),
    )

    mapper.compute_3d_occupancy_information()
    mapper.write_3d_occupancy_grid(str(tmp_path))

    loaded = o3d.t.io.read_point_cloud(str(tmp_path / "occupancy_pcd.ply"))
    with pytest.raises(KeyError):
        loaded.point["normals"]
    assert not (tmp_path / "occupancy_grid_bonxai.bin").exists()
