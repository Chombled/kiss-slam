from types import SimpleNamespace

import numpy as np
import pytest

import kiss_slam.pipeline as pipeline_module
from kiss_icp.scan import LidarScan
from kiss_slam.pipeline import (
    MAP_CLOSURES_RELEASE_FLOOR,
    RefuseScansCompatibilityError,
    SlamPipeline,
)


class _Dataset:
    def __init__(self, scans):
        self._scans = scans
        self.reset_calls = 0

    def __getitem__(self, idx):
        return self._scans[idx]

    def reset(self):
        self.reset_calls += 1


def test_refuse_scans_requires_ground_alignment_api(monkeypatch):
    monkeypatch.setattr(pipeline_module, "_installed_map_closures_version", lambda: "2.0.1")

    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=object()))

    with pytest.raises(RefuseScansCompatibilityError) as exc_info:
        pipeline._validate_refuse_scans_support()

    message = str(exc_info.value)
    assert MAP_CLOSURES_RELEASE_FLOOR in message
    assert "Install a newer build from the upstream MapClosures repository" in message
    assert "local map PLY exports" in message
    assert "without `--refuse-scans`" in message


def test_global_mapping_integrates_frames_when_ground_alignment_api_exists(monkeypatch, tmp_path):
    class Detector:
        def get_ground_alignment_from_id(self, idx):
            assert idx == 0
            return np.eye(4)

    class Preprocessor:
        def __init__(self):
            self.calls = []

        def preprocess(self, scan, delta):
            self.calls.append((scan, delta.copy()))
            return scan

    class OccupancyMapper:
        last_instance = None

        def __init__(self, config):
            self.config = config
            self.integrated = []
            self.computed_3d = False
            self.computed_2d = False
            self.written_3d = None
            self.written_2d = None
            OccupancyMapper.last_instance = self

        def integrate_frame(self, frame, pose):
            self.integrated.append((frame, pose.copy()))

        def compute_3d_occupancy_information(self):
            self.computed_3d = True

        def compute_2d_occupancy_information(self):
            self.computed_2d = True

        def write_3d_occupancy_grid(self, output_dir):
            self.written_3d = output_dir

        def write_2d_occupancy_grid(self, output_dir):
            self.written_2d = output_dir

    dataset = _Dataset(
        [
            LidarScan(
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([1.0], dtype=np.float32),
            ),
            LidarScan(
                points=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([2.0], dtype=np.float32),
            ),
        ]
    )
    preprocessor = Preprocessor()

    monkeypatch.setattr(pipeline_module, "OccupancyGridMapper", OccupancyMapper)
    monkeypatch.setattr(pipeline_module, "trange", lambda start, stop, **kwargs: range(start, stop))

    import kiss_icp.preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "get_preprocessor", lambda config: preprocessor)

    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=Detector()))
    pipeline._dataset = dataset
    pipeline._first = 0
    pipeline._last = 2
    pipeline.poses = np.stack((np.eye(4), np.eye(4)))
    pipeline.config = object()
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=True,
            export_2d_map=True,
            export_bonxai_volume=True,
        )
    )
    pipeline.results_dir = str(tmp_path)
    pipeline._next = lambda idx: dataset[idx]

    pipeline._validate_refuse_scans_support()
    pipeline._global_mapping()

    mapper = OccupancyMapper.last_instance
    assert mapper is not None
    assert dataset.reset_calls == 1
    assert len(preprocessor.calls) == 2
    assert len(mapper.integrated) == 2
    assert all(frame.has_intensity for frame, _ in mapper.integrated)
    assert np.allclose(mapper.integrated[0][0].intensities, [1.0])
    assert np.allclose(mapper.integrated[1][0].intensities, [2.0])
    assert mapper.computed_3d is True
    assert mapper.computed_2d is True
    assert mapper.written_3d == str(tmp_path / "occupancy_grid")
    assert mapper.written_2d == str(tmp_path / "occupancy_grid" / "map2d")


def test_global_mapping_writes_2d_outputs_when_enabled(monkeypatch, tmp_path):
    class Detector:
        def get_ground_alignment_from_id(self, idx):
            assert idx == 0
            return np.eye(4)

    class Preprocessor:
        def preprocess(self, scan, delta):
            return scan

    class OccupancyMapper:
        last_instance = None

        def __init__(self, config):
            self.computed_3d = False
            self.computed_2d = False
            self.written_3d = None
            self.written_2d = None
            OccupancyMapper.last_instance = self

        def integrate_frame(self, frame, pose):
            pass

        def compute_3d_occupancy_information(self):
            self.computed_3d = True

        def compute_2d_occupancy_information(self):
            self.computed_2d = True

        def write_3d_occupancy_grid(self, output_dir):
            self.written_3d = output_dir

        def write_2d_occupancy_grid(self, output_dir):
            self.written_2d = output_dir

    dataset = _Dataset(
        [
            LidarScan(
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([1.0], dtype=np.float32),
            ),
            LidarScan(
                points=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([2.0], dtype=np.float32),
            ),
        ]
    )

    monkeypatch.setattr(pipeline_module, "OccupancyGridMapper", OccupancyMapper)
    monkeypatch.setattr(pipeline_module, "trange", lambda start, stop, **kwargs: range(start, stop))

    import kiss_icp.preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "get_preprocessor", lambda config: Preprocessor())

    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=Detector()))
    pipeline._dataset = dataset
    pipeline._first = 0
    pipeline._last = 2
    pipeline.poses = np.stack((np.eye(4), np.eye(4)))
    pipeline.config = object()
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=True,
            export_2d_map=True,
            export_bonxai_volume=False,
        )
    )
    pipeline.results_dir = str(tmp_path)
    pipeline._next = lambda idx: dataset[idx]

    pipeline._global_mapping()

    mapper = OccupancyMapper.last_instance
    assert mapper is not None
    assert mapper.computed_3d is True
    assert mapper.computed_2d is True
    assert mapper.written_3d == str(tmp_path / "occupancy_grid")
    assert mapper.written_2d == str(tmp_path / "occupancy_grid" / "map2d")


def test_global_mapping_skips_2d_outputs_when_disabled(monkeypatch, tmp_path):
    class Detector:
        def get_ground_alignment_from_id(self, idx):
            assert idx == 0
            return np.eye(4)

    class Preprocessor:
        def __init__(self):
            self.calls = []

        def preprocess(self, scan, delta):
            self.calls.append((scan, delta.copy()))
            return scan

    class OccupancyMapper:
        last_instance = None

        def __init__(self, config):
            self.integrated = []
            self.computed_3d = False
            self.computed_2d = False
            self.written_3d = None
            self.written_2d = None
            OccupancyMapper.last_instance = self

        def integrate_frame(self, frame, pose):
            self.integrated.append((frame, pose.copy()))

        def compute_3d_occupancy_information(self):
            self.computed_3d = True

        def compute_2d_occupancy_information(self):
            self.computed_2d = True

        def write_3d_occupancy_grid(self, output_dir):
            self.written_3d = output_dir

        def write_2d_occupancy_grid(self, output_dir):
            self.written_2d = output_dir

    dataset = _Dataset(
        [
            LidarScan(
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([1.0], dtype=np.float32),
            ),
            LidarScan(
                points=np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([2.0], dtype=np.float32),
            ),
        ]
    )
    preprocessor = Preprocessor()

    monkeypatch.setattr(pipeline_module, "OccupancyGridMapper", OccupancyMapper)
    monkeypatch.setattr(pipeline_module, "trange", lambda start, stop, **kwargs: range(start, stop))

    import kiss_icp.preprocess as preprocess_module

    monkeypatch.setattr(preprocess_module, "get_preprocessor", lambda config: preprocessor)

    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=Detector()))
    pipeline._dataset = dataset
    pipeline._first = 0
    pipeline._last = 2
    pipeline.poses = np.stack((np.eye(4), np.eye(4)))
    pipeline.config = object()
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=True,
            export_2d_map=False,
            export_bonxai_volume=False,
        )
    )
    pipeline.results_dir = str(tmp_path)
    pipeline._next = lambda idx: dataset[idx]

    pipeline._global_mapping()

    mapper = OccupancyMapper.last_instance
    assert mapper is not None
    assert dataset.reset_calls == 1
    assert len(preprocessor.calls) == 2
    assert len(mapper.integrated) == 2
    assert mapper.computed_3d is True
    assert mapper.computed_2d is False
    assert mapper.written_3d == str(tmp_path / "occupancy_grid")
    assert mapper.written_2d is None


def test_global_mapping_skips_rescan_when_all_outputs_are_disabled(monkeypatch, tmp_path):
    class Detector:
        def get_ground_alignment_from_id(self, idx):
            raise AssertionError("ground alignment should not be requested")

    class OccupancyMapper:
        def __init__(self, config):
            raise AssertionError("occupancy mapper should not be constructed")

    dataset = _Dataset(
        [
            LidarScan(
                points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                timestamps=np.array([]),
                intensities=np.array([1.0], dtype=np.float32),
            )
        ]
    )

    monkeypatch.setattr(pipeline_module, "OccupancyGridMapper", OccupancyMapper)

    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=Detector()))
    pipeline._dataset = dataset
    pipeline._first = 0
    pipeline._last = 1
    pipeline.poses = np.stack((np.eye(4),))
    pipeline.config = object()
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=False,
            export_2d_map=False,
            export_bonxai_volume=False,
        )
    )
    pipeline.results_dir = str(tmp_path)

    pipeline._global_mapping()

    assert dataset.reset_calls == 0


def test_refuse_scans_compatibility_check_is_skipped_when_all_outputs_are_disabled():
    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = True
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=object()))
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=False,
            export_2d_map=False,
            export_bonxai_volume=False,
        )
    )

    pipeline._validate_refuse_scans_support()


def test_refuse_scans_compatibility_check_is_skipped_when_flag_is_disabled():
    pipeline = SlamPipeline.__new__(SlamPipeline)
    pipeline.refuse_scans = False
    pipeline.kiss_slam = SimpleNamespace(closer=SimpleNamespace(detector=object()))
    pipeline.slam_config = SimpleNamespace(
        occupancy_mapper=SimpleNamespace(
            export_ply=True,
            export_2d_map=True,
            export_bonxai_volume=True,
        )
    )

    pipeline._validate_refuse_scans_support()
