#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ==============================================================================
# Master 1-Click Pipeline for Cobot Vision-Language-Action (VLA) Fine-Tuning
# Supports Models: pi0, pi0.5, smolvla, act
# Output Logs & Models Directory : /workspace/core/logs/vla
# ==============================================================================

set -e

MODEL_TYPE="${1:-pi0}"
DATASET_PATH="${2:-/workspace/core/data/vla/cobot_vla_dataset.h5}"
OUTPUT_DIR="${3:-/workspace/core/logs/vla}"
PRETRAINED_HUB="${4:-lerobot/pi0_ur5}"

echo "=================================================================="
echo "      LAUNCHING COBOT VLA FINE-TUNING & INFERENCE PIPELINE        "
echo "=================================================================="
echo " Model Architecture  : ${MODEL_TYPE}"
echo " Pretrained Hub Weights: ${PRETRAINED_HUB}"
echo " Dataset Path        : ${DATASET_PATH}"
echo " Output Models Dir   : ${OUTPUT_DIR}"
echo "=================================================================="

# Step 1: Inspect Dataset
echo "[Step 1/3] Inspecting Cobot VLA Dataset..."
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/dataset_inspector.py --dataset "${DATASET_PATH}"

# Step 2: Fine-Tune Pretrained VLA Model
echo "[Step 2/3] Fine-Tuning '${MODEL_TYPE}' on Cobot Joint Demonstrations..."
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/source/cobot/vla/train_vla.py \
    --model "${MODEL_TYPE}" \
    --pretrained_hub "${PRETRAINED_HUB}" \
    --dataset "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 5

# Step 3: Run Closed-Loop Inference
echo "[Step 3/3] Running Closed-Loop VLA Inference in Simulation / ROS2..."
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/source/cobot/vla/inference_vla.py \
    --model "${MODEL_TYPE}" \
    --ckpt "${OUTPUT_DIR}/${MODEL_TYPE}_cobot_policy.pt" \
    --prompt "reach and touch target red object"

echo "=================================================================="
echo " SUCCESS! Cobot VLA Pipeline Completed cleanly."
echo " Saved model weights -> ${OUTPUT_DIR}/${MODEL_TYPE}_cobot_policy.pt"
echo "=================================================================="
