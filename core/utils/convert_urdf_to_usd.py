#!/usr/bin/env python3
"""Utility script for converting URDF robot asset files to OpenUSD (.usd/.usda) assets.

Usage (inside container):
  /isaac-sim/python.sh core/utils/convert_urdf_to_usd.py --urdf <path_to_urdf> --output <path_to_usd>
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert URDF robot description to OpenUSD format.")
    parser.add_argument("--urdf", type=str, required=False, help="Path to input URDF file.")
    parser.add_argument("--output", type=str, required=False, help="Path to output USD file.")
    args = parser.parse_args()

    print("[INFO] URDF to USD converter utility initialized.")
    if not args.urdf or not args.output:
        print("Usage: python core/utils/convert_urdf_to_usd.py --urdf <input.urdf> --output <output.usd>")
        sys.exit(0)

    if not os.path.exists(args.urdf):
        raise FileNotFoundError(f"Input URDF file not found: {args.urdf}")

    print(f"[INFO] Converting '{args.urdf}' -> '{args.output}'...")
    # Core IsaacLab URDF converter entrypoint
    try:
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": True})
        simulation_app = app_launcher.app
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        conv_cfg = UrdfConverterCfg(asset_path=args.urdf, usd_dir=os.path.dirname(args.output))
        converter = UrdfConverter(conv_cfg)
        usd_path = converter.usd_path
        print(f"[INFO] Successfully converted URDF to USD: {usd_path}")
        simulation_app.close()
    except Exception as e:
        print(f"[ERROR] Could not run Isaac Sim URDF converter automatically: {e}")


if __name__ == "__main__":
    main()
