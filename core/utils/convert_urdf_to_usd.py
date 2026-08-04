#!/usr/bin/env python3
"""Convert a URDF robot asset into a USD file using Isaac Lab's URDF importer.

This is a thin, headless CLI wrapper around :class:`isaaclab.sim.converters.UrdfConverter`
(``UrdfConverterCfg``). It produces a USD file named exactly after ``--output`` and bakes in
the requested joint-drive configuration, mirroring the official Isaac Lab utility
``isaaclab/scripts/tools/convert_urdf.py``.

The conversion requires a running Omniverse Kit app, so it must be executed inside the container
and always launches headless (``AppLauncher`` sets ``create_new_stage=False`` + ``hide_ui=True``
in headless mode, which also avoids the viewport-wait hang seen with a bare
``SimulationApp({"headless": True})``).

Usage (inside container):
  /isaac-sim/python.sh core/utils/convert_urdf_to_usd.py --urdf <path_to_urdf> --output <path_to_usd>
      [--joint-stiffness 10.0] [--joint-damping 1.0] [--joint-target-type velocity]
      [--fix-base] [--merge-joints]

Exit code is 0 on success and non-zero on any failure (no exceptions are swallowed).
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert URDF robot description to OpenUSD format.")
parser.add_argument("--urdf", type=str, required=True, help="Path to input URDF file.")
parser.add_argument("--output", type=str, required=True, help="Path to store the output USD file.")
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=100.0,
    help="Stiffness of the joint drive (default: 100.0).",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=1.0,
    help="Damping of the joint drive (default: 1.0).",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="position",
    choices=["position", "velocity", "none"],
    help="Control mode for the joint drive (default: 'position').",
)
parser.add_argument(
    "--fix-base",
    action="store_true",
    default=False,
    help="Fix the base to where it is imported (default: False).",
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=False,
    help="Consolidate links connected by fixed joints (default: False).",
)
args_cli = parser.parse_args()


def main() -> None:
    urdf_path = os.path.abspath(args_cli.urdf)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"Input URDF file not found: {urdf_path}")

    dest_path = os.path.abspath(args_cli.output)
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    simulation_app = None
    try:
        app_launcher = AppLauncher({"headless": True})
        simulation_app = app_launcher.app

        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        conv_cfg = UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=dest_dir,
            usd_file_name=os.path.basename(dest_path),
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_joints,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args_cli.joint_stiffness,
                    damping=args_cli.joint_damping,
                ),
                target_type=args_cli.joint_target_type,
            ),
        )

        converter = UrdfConverter(conv_cfg)
        print(f"[INFO] Generated USD file: {converter.usd_path}")
        if not os.path.isfile(converter.usd_path):
            raise RuntimeError(f"Converter reported USD path but file does not exist: {converter.usd_path}")
        print(f"[INFO] Successfully converted '{urdf_path}' -> '{converter.usd_path}'")
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] Conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)
