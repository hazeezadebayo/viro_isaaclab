# Cobot Vision-Language-Action (VLA) Helper Suite (`third_party/vla_tools/`)

An end-to-end, professor-level guide explaining how to inspect, fine-tune, and evaluate **Vision-Language-Action (VLA)** models ($\pi_0$, $\pi_{0.5}$, SmolVLA, ACT) using 1-click execution scripts.

---

## 1. First Principles: The VLA Tooling Workflow

```
  Multi-Modal Dataset (.json / .h5)
         │
         ▼
 ┌────────────────────────┐
 │ 1. dataset_inspector   │ ───> Inspects joint angles, images, & prompts
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 2. train_vla.py        │ ───> Fine-tunes pre-trained weights (lerobot/pi0_ur5)
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ 3. inference_vla.py    │ ───> Executes closed-loop evaluation in IsaacLab & ROS2
 └────────────────────────┘
```

---

## 2. 1-Click Execution Examples

### Example 1: Run Full Pipeline (1-Click)
```bash
bash /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/run_vla_pipeline.sh \
    pi0 \
    /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json
```

### Example 2: Inspect Dataset Telemetry
```bash
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/dataset_inspector.py \
    --dataset /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json
```

---

## 3. Sub-Module File Breakdown

- **`run_vla_pipeline.sh`**: Master 1-click execution script that orchestrates dataset inspection, fine-tuning pretrained weights, and running simulation inference.
- **`dataset_inspector.py`**: CLI telemetry analyzer displaying dataset frame resolution, total trajectory steps, language prompts, and 6-DOF joint angle ranges.
