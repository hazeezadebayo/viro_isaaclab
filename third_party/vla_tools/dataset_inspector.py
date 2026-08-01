#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dataset Inspector Tool for Cobot VLA Datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys


def inspect_vla_dataset(dataset_path: str):
    """Parses and prints a detailed breakdown of Cobot VLA dataset files (.json, .h5, .npz)."""
    print(f"\n==================================================================================")
    print(f"                      COBOT VLA DATASET INSPECTOR REPORT                          ")
    print(f"==================================================================================")
    print(f" Dataset File Path : {os.path.abspath(dataset_path)}")

    if not os.path.exists(dataset_path):
        print(f" [ERROR] File not found: '{dataset_path}'")
        return

    if dataset_path.endswith(".json"):
        with open(dataset_path, "r") as f:
            data = json.load(f)

        print(f" Target Robot      : {data.get('robot', 'Cobot UR5')}")
        print(f" Total Episodes    : {data.get('num_episodes', len(data.get('episodes', [])))}")

        for ep in data.get("episodes", []):
            ep_id = ep.get("episode_id", "unknown")
            prompt = ep.get("language_prompt", "")
            joints = ep.get("joint_names", [])
            traj = ep.get("trajectory", [])
            print(f"\n --- Episode ID: {ep_id} ---")
            print(f"     Language Prompt : '{prompt}'")
            print(f"     Cobot Joints    : {joints}")
            print(f"     Trajectory Steps: {len(traj)}")
            if len(traj) > 0:
                print(f"     Initial Joints  : {traj[0].get('joint_pos')}")
                print(f"     Final Target    : {traj[-1].get('joint_pos')}")
                print(f"     Camera Resolution: {traj[0].get('image_meta', {}).get('width')}x{traj[0].get('image_meta', {}).get('height')}")

    elif dataset_path.endswith(".h5") or dataset_path.endswith(".hdf5"):
        try:
            import h5py
            with h5py.File(dataset_path, "r") as f:
                print(f" Total HDF5 Groups : {len(f.keys())}")
                for key in f.keys():
                    grp = f[key]
                    prompt = grp.attrs.get("language_prompt", "N/A")
                    images = grp["image"]
                    qpos = grp["joint_pos"]
                    print(f"\n --- {key} ---")
                    print(f"     Language Prompt : '{prompt}'")
                    print(f"     Image Shape     : {images.shape} ({images.dtype})")
                    print(f"     Joint Pos Shape : {qpos.shape} ({qpos.dtype})")
        except ImportError:
            print(" [NOTICE] Install 'h5py' package to inspect HDF5 binary files.")

    print(f"\n==================================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Cobot VLA Dataset.")
    parser.add_argument("--dataset", type=str, default="/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json")
    args = parser.parse_args()

    inspect_vla_dataset(args.dataset)
