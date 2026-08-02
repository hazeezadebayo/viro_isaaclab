# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export headless Isaac Lab policy rollouts as animated USD (.usda) stages and automatic MP4 video clips.

The :class:`UsdTrajectoryExporter` records per-step world poses for every articulation and
rigid object in the scene and bakes them into an animated USD stage.

The :class:`PeriodicUsdExporterWrapper` wraps an Isaac Lab environment to automatically capture
Y seconds of rollout trajectory every X seconds of simulation time during training or play,
baking `trajectory_t1.usda` and immediately running the pure-Python `usd_to_mp4` converter to produce `trajectory_t1.mp4`.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from core.utils.usd_to_mp4 import convert_usd_to_mp4

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)

__all__ = ["UsdTrajectoryExporter", "PeriodicUsdExporterWrapper"]


class UsdTrajectoryExporter:
    """Records robot trajectories and bakes them into an animated, self-contained USD stage."""

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        out_dir: str = "/workspace/core/logs/usd",
        fmt: str = "usda",
        stage: Any = None,
        env_index: int = 0,
        root_path: str = "/World",
        filename: str = "trajectory",
    ) -> None:
        self.env = env
        self.fmt = str(fmt).lstrip(".").lower()
        if self.fmt not in ("usda", "usdc", "usd"):
            raise ValueError(f"Unsupported USD format: {self.fmt!r} (expected 'usda', 'usdc' or 'usd')")
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.filename = filename
        self._stage = stage
        self.root_path = str(root_path).rstrip("/")
        self.env_index = int(env_index)
        self.step_dt = float(getattr(env, "step_dt", 1.0 / 60.0))

        self._entities: list[tuple[str, str, str, Any, list[str] | None]] = []
        scene = getattr(env, "scene", None)
        if scene is not None:
            for name, artic in getattr(scene, "articulations", {}).items():
                body_names = list(getattr(artic, "body_names", []))
                self._entities.append(("articulation", name, self._root_prim_path(artic), artic, body_names))
            for name, obj in getattr(scene, "rigid_objects", {}).items():
                self._entities.append(("rigid_object", name, self._root_prim_path(obj), obj, None))
        self._frames: list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in self._entities]

        logger.info(
            "UsdTrajectoryExporter: %d entity(s) (env %d), step_dt=%.5fs, format=%s",
            len(self._entities),
            self.env_index,
            self.step_dt,
            self.fmt,
        )

    @staticmethod
    def _root_prim_path(asset: Any) -> str:
        """Resolve the root prim path of an articulation or rigid object."""
        view = getattr(asset, "root_physx_view", None)
        if view is not None:
            paths = getattr(view, "prim_paths", None)
            if paths:
                return paths[0]
        cfg = getattr(asset, "cfg", None)
        if cfg is not None:
            prim_path = getattr(cfg, "prim_path", None)
            if prim_path:
                return prim_path
        return str(asset)

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        """Coerce a torch tensor / numpy array into a float64 2D numpy array."""
        if hasattr(value, "cpu"):
            value = value.cpu()
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def capture(self) -> None:
        """Record the current world poses of every tracked entity."""
        for i, (kind, _name, _root, asset, _body_names) in enumerate(self._entities):
            if kind == "articulation":
                pos = self._as_numpy(asset.data.body_pos_w[self.env_index])
                quat = self._as_numpy(asset.data.body_quat_w[self.env_index])
            else:
                pos = self._as_numpy(asset.data.root_pos_w[self.env_index])
                quat = self._as_numpy(asset.data.root_quat_w[self.env_index])
            self._frames[i].append((pos, quat))

    def finalize(self) -> str:
        """Bake the recorded poses into an animated USD file."""
        from pxr import Gf, Sdf, Usd, UsdGeom

        stage = self._stage if self._stage is not None else self._resolve_live_stage()

        flat_layer = stage.Flatten()
        flat = flat_layer if isinstance(flat_layer, Usd.Stage) else Usd.Stage.Open(flat_layer)

        env_prefix = f"{self.root_path}/envs/env_{self.env_index}"
        for prim in list(flat.Traverse()):
            path = str(prim.GetPath())
            if path.startswith(f"{self.root_path}/envs/env_") and not path.startswith(env_prefix):
                flat.RemovePrim(prim.GetPath())

        body_prim_cache: dict[str, dict[str, Any]] = {}
        for (kind, name, root_path, _asset, body_names), frames in zip(self._entities, self._frames):
            if not frames:
                continue
            if kind == "articulation":
                self._author_articulation(flat, name, root_path, body_names, frames, body_prim_cache)
            else:
                self._author_rigid_object(flat, name, root_path, frames)

        n_frames = max(len(f) for f in self._frames) if self._frames else 0
        flat.SetTimeCodesPerSecond(1.0 / self.step_dt)
        flat.SetStartTimeCode(0)
        flat.SetEndTimeCode(max(0, n_frames - 1))

        out_path = os.path.join(self.out_dir, f"{self.filename}.{self.fmt}")
        if not flat.GetRootLayer().Export(out_path):
            raise RuntimeError(f"Failed to export USD to {out_path}")
        logger.info("Exported animated USD (%d frame(s)) to %s", n_frames, out_path)
        return out_path

    def _resolve_live_stage(self) -> Any:
        """Resolve the live USD stage from the running Isaac Sim app."""
        try:
            import omni.usd
        except ImportError as e:
            raise RuntimeError(
                "No USD stage available. Pass `stage=` when running outside the Isaac Sim app."
            ) from e
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No live USD stage found (is the simulation app running?).")
        return stage

    @staticmethod
    def _iter_prim_tree(prim: Any) -> Any:
        yield prim
        for child in prim.GetChildren():
            yield from UsdTrajectoryExporter._iter_prim_tree(child)

    def _static_world(self, flat: Any, prim: Any) -> Any:
        from pxr import Gf, Usd, UsdGeom

        if not prim.IsValid():
            return Gf.Matrix4d().SetIdentity()
        return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode(0))

    def _find_robot_root(self, flat: Any, root_body_path: str, body_names: list[str]) -> Any:
        prim = flat.GetPrimAtPath(root_body_path)
        wanted = set(body_names)
        while prim.IsValid() and str(prim.GetPath()) != "/":
            sub_names = [q.GetName() for q in self._iter_prim_tree(prim)]
            if all(n in sub_names for n in wanted):
                return prim
            prim = prim.GetParent()
        return prim

    def _collect_body_prims(
        self, flat: Any, root_path: str, body_names: list[str], cache: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if root_path in cache:
            return cache[root_path]
        root = flat.GetPrimAtPath(root_path)
        wanted = set(body_names)
        found: dict[str, Any] = {}
        if root.IsValid():
            for child in root.GetAllChildren():
                for prim in self._iter_prim_tree(child):
                    if prim.GetName() in wanted:
                        found[prim.GetName()] = prim
        cache[root_path] = found
        return found

    def _parent_pose(
        self,
        flat: Any,
        parent: Any,
        frames: list[tuple[np.ndarray, np.ndarray]],
        body_names: list[str],
        body_prims: dict[str, Any],
        index: int,
        parent_mat_cache: dict[str, Any],
    ) -> Any:
        if parent.IsValid() and parent.GetName() in body_prims and parent.GetName() in body_names:
            pidx = body_names.index(parent.GetName())
            return (frames[index][0][pidx], frames[index][1][pidx])
        key = str(parent.GetPath())
        if key not in parent_mat_cache:
            parent_mat_cache[key] = self._static_world(flat, parent)
        return parent_mat_cache[key]

    def _local_of(self, child_world_pos: Any, child_world_q: np.ndarray, parent_mat_or_pose: Any) -> Any:
        from pxr import Gf

        child_q = Gf.Quatd(child_world_q[0], Gf.Vec3d(*child_world_q[1:]))
        if isinstance(parent_mat_or_pose, tuple):
            ppos, pquat_arr = parent_mat_or_pose
            pquat = Gf.Quatd(pquat_arr[0], Gf.Vec3d(*pquat_arr[1:]))
            local_pos = pquat.GetInverse().Transform(child_world_pos - Gf.Vec3d(*ppos))
            local_q = pquat.GetInverse() * child_q
        else:
            matrix = parent_mat_or_pose
            local_pos = matrix.GetInverse().Transform(child_world_pos)
            pquat = matrix.ExtractRotation().GetQuat()
            local_q = pquat.GetInverse() * child_q
        return local_pos, local_q

    def _author_articulation(
        self,
        flat: Any,
        name: str,
        root_body_path: str,
        body_names: list[str],
        frames: list[tuple[np.ndarray, np.ndarray]],
        body_prim_cache: dict[str, dict[str, Any]],
    ) -> None:
        from pxr import Gf, Usd, UsdGeom

        robot_root = self._find_robot_root(flat, root_body_path, body_names)
        if not robot_root.IsValid():
            logger.warning("[%s] articulation root not found at %s; skipping", name, root_body_path)
            return
        body_prims = self._collect_body_prims(flat, str(robot_root.GetPath()), body_names, body_prim_cache)
        parent_mat_cache: dict[str, Any] = {}
        for bname, prim in body_prims.items():
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            top = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            orop = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            parent = prim.GetParent()
            idx = body_names.index(bname)
            for i, (bp, bq) in enumerate(frames):
                parent_pose = self._parent_pose(flat, parent, frames, body_names, body_prims, i, parent_mat_cache)
                local_pos, local_q = self._local_of(Gf.Vec3d(*bp[idx]), bq[idx], parent_pose)
                top.Set(local_pos, Usd.TimeCode(i))
                orop.Set(local_q, Usd.TimeCode(i))

    def _author_rigid_object(
        self, flat: Any, name: str, root_path: str, frames: list[tuple[np.ndarray, np.ndarray]]
    ) -> None:
        from pxr import Gf, Usd, UsdGeom

        prim = flat.GetPrimAtPath(root_path)
        if not prim.IsValid():
            logger.warning("[%s] rigid object prim not found at %s; skipping", name, root_path)
            return
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        top = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        orop = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        parent_mat = self._static_world(flat, prim.GetParent())
        for i, (rp, rq) in enumerate(frames):
            rp = np.asarray(rp).reshape(-1)
            rq = np.asarray(rq).reshape(-1)
            local_pos, local_q = self._local_of(Gf.Vec3d(*rp), rq, parent_mat)
            top.Set(local_pos, Usd.TimeCode(i))
            orop.Set(local_q, Usd.TimeCode(i))


class PeriodicUsdExporterWrapper:
    """Wraps an Isaac Lab environment to automatically export Y seconds of trajectory every X seconds.

    Automatically bakes `trajectory_t1.usda`, `trajectory_t2.usda` and immediately converts each to `.mp4`.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        out_dir: str = "/workspace/core/logs/usd",
        record_interval_s: float = 1800.0,  # X: seconds between export starts (default 30 mins)
        clip_length_s: float = 10.0,        # Y: seconds of rollout duration per capture (default 10s)
        convert_to_mp4: bool = True,
    ):
        self.env = env
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.record_interval_s = record_interval_s
        self.clip_length_s = clip_length_s
        self.convert_to_mp4 = convert_to_mp4

        self.step_dt = float(getattr(env.unwrapped, "step_dt", 1.0 / 60.0))
        self.clip_steps = max(1, int(self.clip_length_s / self.step_dt))
        self.interval_steps = max(self.clip_steps, int(self.record_interval_s / self.step_dt))

        self.step_counter = 0
        self.clip_count = 0
        self.recording = False
        self._current_exporter: UsdTrajectoryExporter | None = None
        self._recorded_in_current_clip = 0

        logger.info(
            f"PeriodicUsdExporterWrapper initialized: Interval X={self.record_interval_s}s ({self.interval_steps} steps), "
            f"Clip Y={self.clip_length_s}s ({self.clip_steps} steps). Output: {self.out_dir}"
        )

    def step(self, action: Any) -> tuple[Any, Any, Any, Any]:
        """Wrap env.step to trigger automatic periodic USD exports."""
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_counter += 1

        # Check if it's time to start a new export clip
        if not self.recording and (self.step_counter % self.interval_steps == 1 or self.step_counter == 1):
            self.clip_count += 1
            filename = f"trajectory_t{self.clip_count}"
            self._current_exporter = UsdTrajectoryExporter(
                self.env.unwrapped,
                out_dir=self.out_dir,
                fmt="usda",
                filename=filename,
            )
            self.recording = True
            self._recorded_in_current_clip = 0
            logger.info(f"[USD Exporter] Starting capture for {filename}.usda ({self.clip_length_s}s)")

        # Capture step if active
        if self.recording and self._current_exporter is not None:
            self._current_exporter.capture()
            self._recorded_in_current_clip += 1

            if self._recorded_in_current_clip >= self.clip_steps:
                usda_path = self._current_exporter.finalize()
                logger.info(f"[USD Exporter] Finalized {usda_path}")
                self.recording = False
                self._current_exporter = None

                # Convert to MP4 immediately
                if self.convert_to_mp4:
                    mp4_path = f"{os.path.splitext(usda_path)[0]}.mp4"
                    try:
                        convert_usd_to_mp4(usda_path, mp4_path, fps=int(1.0 / self.step_dt))
                        logger.info(f"[USD Exporter] Automatically rendered matching MP4: {mp4_path}")
                    except Exception as e:
                        logger.error(f"[USD Exporter] MP4 conversion failed for {usda_path}: {e}")

        return obs, rew, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to underlying environment."""
        return getattr(self.env, name)
