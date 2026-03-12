import numpy as np
import open3d as o3d

from kiss_slam.local_map_graph import LocalMapGraph
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
