# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for resolving the pre-trained AMR locomotion policy path."""

from __future__ import annotations

import os


def get_locomotion_policy_path() -> str:
    """Dynamically resolve the AMR locomotion policy path.

    Searches for ``exported/policy.pt`` under the most recent run folders of the RSL-RL
    experiment ``amr_locomotion`` and falls back to ``pretrained/policy.pt``.
    """
    experiment_name = "amr_locomotion"
    base_log_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "logs",
            "rsl_rl",
            experiment_name,
        )
    )

    if os.path.exists(base_log_dir):
        run_dirs = [
            os.path.join(base_log_dir, d)
            for d in os.listdir(base_log_dir)
            if os.path.isdir(os.path.join(base_log_dir, d)) and d != "pretrained"
        ]
        if run_dirs:
            run_dirs.sort()
            for latest_run_dir in reversed(run_dirs):
                exported_path = os.path.join(latest_run_dir, "exported", "policy.pt")
                if os.path.exists(exported_path):
                    print(f"[INFO] Resolved AMR locomotion policy dynamically to: {exported_path}")
                    return exported_path

    fallback_path = os.path.join(base_log_dir, "pretrained", "policy.pt")
    print(f"[INFO] Fallback AMR locomotion policy resolved to: {fallback_path}")
    return fallback_path
