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
import os
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
from kiss_icp.pipeline import OdometryPipeline
from kiss_icp.scan import coerce_scan
from tqdm import tqdm, trange

from kiss_slam.config import load_config
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.slam import KissSLAM
from kiss_slam.tools.visualizer import RegistrationVisualizer, StubVisualizer


MAP_CLOSURES_RELEASE_FLOOR = "2.0.2"


class RefuseScansCompatibilityError(RuntimeError):
    """Raised when the installed MapClosures build cannot support final fusion."""


def _installed_map_closures_version() -> Optional[str]:
    for distribution_name in ("map_closures", "map-closures"):
        try:
            return metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            continue
    return None


class SlamPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        config_file: Optional[Path] = None,
        visualize: bool = False,
        n_scans: int = -1,
        jump: int = 0,
        refuse_scans: bool = False,
    ):
        super().__init__(dataset=dataset, config=None, n_scans=n_scans, jump=jump)
        self.slam_config = load_config(config_file)
        self.config = self.slam_config.kiss_icp_config()
        self.visualize = visualize
        self.kiss_slam = KissSLAM(self.slam_config)
        self.visualizer = RegistrationVisualizer() if self.visualize else StubVisualizer()
        self.refuse_scans = refuse_scans
        self._validate_refuse_scans_support()

    def _occupancy_outputs_enabled(self) -> bool:
        occupancy_config = self.slam_config.occupancy_mapper
        return any(
            (
                getattr(occupancy_config, "export_ply", True),
                getattr(occupancy_config, "export_2d_map", True),
                getattr(occupancy_config, "export_bonxai_volume", True),
            )
        )

    def _validate_refuse_scans_support(self):
        if not self.refuse_scans or not self._occupancy_outputs_enabled():
            return

        detector = self.kiss_slam.closer.detector
        if hasattr(detector, "get_ground_alignment_from_id"):
            return

        detected_version = _installed_map_closures_version()
        version_suffix = (
            f"Detected map_closures version: {detected_version}."
            if detected_version
            else "The installed map_closures version could not be determined."
        )
        raise RefuseScansCompatibilityError(
            "`--refuse-scans` requires `MapClosures.get_ground_alignment_from_id`, "
            f"but the installed MapClosures build does not expose it. {version_suffix} "
            f"The current release floor is map_closures>={MAP_CLOSURES_RELEASE_FLOOR}, but "
            "some published 2.0.2 builds still predate this API. Install a newer build from "
            "the upstream MapClosures repository, or rerun without `--refuse-scans` if you "
            "only need the standard SLAM outputs and local map PLY exports; final "
            "occupancy/global fusion depends on the newer ground-alignment API."
        )

    def run(self):
        self._run_pipeline()
        self._run_evaluation()
        self._evaluate_closures()
        self._create_output_dir()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        self._write_graph()
        self._write_closures()
        self._write_local_maps()
        self._global_mapping()
        return self.results

    def _run_pipeline(self):
        for idx in trange(self._first, self._last, unit=" frames", dynamic_ncols=True):
            scan = self._next(idx)
            start_time = time.perf_counter_ns()
            self.kiss_slam.process_scan(scan)
            self.times[idx - self._first] = time.perf_counter_ns() - start_time
            self.visualizer.update(self.kiss_slam)
        self.kiss_slam.generate_new_node()
        self.kiss_slam.local_map_graph.erase_last_local_map()
        self.poses, self.pose_graph = self.kiss_slam.fine_grained_optimization()
        self.poses = np.array(self.poses)

    def _global_mapping(self):
        if self.refuse_scans:
            if not self._occupancy_outputs_enabled():
                return

            from kiss_icp.preprocess import get_preprocessor

            occupancy_config = self.slam_config.occupancy_mapper
            if hasattr(self._dataset, "reset"):
                self._dataset.reset()
            ref_ground_alignment = self.kiss_slam.closer.detector.get_ground_alignment_from_id(0)
            deskewing_deltas = np.vstack(
                (
                    np.eye(4)[None],
                    np.eye(4)[None],
                    np.linalg.inv(self.poses[:-2]) @ self.poses[1:-1],
                )
            )
            preprocessor = get_preprocessor(self.config)
            occupancy_grid_mapper = OccupancyGridMapper(self.slam_config.occupancy_mapper)
            print("KissSLAM| Computing Occupancy Grid")
            for idx in trange(self._first, self._last, unit=" frames", dynamic_ncols=True):
                scan = self._next(idx)
                deskewed_scan = preprocessor.preprocess(scan, deskewing_deltas[idx])
                occupancy_grid_mapper.integrate_frame(
                    deskewed_scan, ref_ground_alignment @ self.poses[idx - self._first]
                )
            occupancy_grid_mapper.compute_3d_occupancy_information()
            occupancy_dir = os.path.join(self.results_dir, "occupancy_grid")
            os.makedirs(occupancy_dir, exist_ok=True)
            if occupancy_config.export_ply or occupancy_config.export_bonxai_volume:
                occupancy_grid_mapper.write_3d_occupancy_grid(occupancy_dir)
            if occupancy_config.export_2d_map:
                occupancy_grid_mapper.compute_2d_occupancy_information()
                occupancy_2d_map_dir = os.path.join(occupancy_dir, "map2d")
                os.makedirs(occupancy_2d_map_dir, exist_ok=True)
                occupancy_grid_mapper.write_2d_occupancy_grid(occupancy_2d_map_dir)

    def _write_local_maps(self):
        local_maps_dir = os.path.join(self.results_dir, "local_maps")
        os.makedirs(local_maps_dir, exist_ok=True)
        self.kiss_slam.optimizer.write_graph(os.path.join(local_maps_dir, "local_map_graph.g2o"))
        plys_dir = os.path.join(local_maps_dir, "plys")
        os.makedirs(plys_dir, exist_ok=True)
        print("KissSLAM| Writing Local Maps on Disk")
        for local_map in tqdm(self.kiss_slam.local_map_graph.local_maps()):
            filename = os.path.join(plys_dir, "{:06d}.ply".format(local_map.id))
            local_map.write(filename)

    def _evaluate_closures(self):
        self.results.append(
            desc="Number of closures found", units="closures", value=len(self.kiss_slam.closures)
        )

    def _write_closures(self):
        import matplotlib.pyplot as plt

        locations = [pose[:3, -1] for pose in self.poses]
        loc_x = [loc[0] for loc in locations]
        loc_y = [loc[1] for loc in locations]
        plt.scatter(loc_x, loc_y, s=0.1, color="black")
        key_poses = self.kiss_slam.get_keyposes()
        for closure in self.kiss_slam.closures:
            i, j = closure
            plt.plot(
                [key_poses[i][0, -1], key_poses[j][0, -1]],
                [key_poses[i][1, -1], key_poses[j][1, -1]],
                color="red",
                linewidth=1,
                markersize=1,
            )
        plt.savefig(os.path.join(self.results_dir, "trajectory.png"), dpi=2000)

    def _write_graph(self):
        self.pose_graph.write_graph(os.path.join(self.results_dir, "trajectory.g2o"))

    def _next(self, idx):
        return coerce_scan(self._dataset[idx])
