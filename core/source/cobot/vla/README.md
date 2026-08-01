# Vision-Language-Action (VLA) Architecture & Fine-Tuning Guide

An end-to-end, professor-level guide answering the 3 fundamental VLA questions: **Pretrained Fine-Tuning vs. Scratch**, **Data Location & Sample Trajectories**, and **1-Click Third-Party Helper Tools**.

---

## 1. Direct Answers to Core Questions

### Q1: Am I training from scratch or downloading a pre-trained model and retraining?
> **YOU ARE FINE-TUNING / RETRAINing PRE-TRAINED WEIGHTS.**
> In modern robotics (HuggingFace LeRobot, Physical Intelligence $\pi_0$, OpenVLA), training vision-language backbones from scratch requires millions of robot interaction hours. 
> 
> Instead, our framework downloads base pre-trained weights (e.g. `lerobot/pi0_ur5` or `lerobot/smolvla`) and **fine-tunes** the action prediction heads using your Cobot dataset! This allows training to converge in **minutes instead of weeks**.

---

### Q2: Where is the data saved, and is there sample Cobot data?
> **DATA LOCATION**: **`/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/`**
> 
> We have provided a ready-to-use sample dataset file:
> **`/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json`**
> 
> It contains **3 real Cobot trajectory data points** covering:
> - **Joint Angles & Velocities**: Exact 6-DOF UR5 joint names (`joint_1` through `joint_6`).
> - **Language Prompts**: *"reach and touch the target red object"*, *"pick up the blue cylinder from table"*, and *"move end-effector to home position"*.
> - **Camera Meta**: $640 \times 480$ RGB camera rendering data.

---

### Q3: Where is the 1-Click Third-Party Helper Suite?
> **HELPER SUITE LOCATION**: **`/home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/`**
> 
> Just like `third_party/retarget_tools/` provides 1-click motion capture processing for Humanoids, `third_party/vla_tools/` provides 1-click VLA execution:

> 
> 1. **`third_party/vla_tools/run_vla_pipeline.sh`**: 1-click master script that inspects data, loads pre-trained VLA weights, fine-tunes the policy, and runs closed-loop inference!
> 2. **`third_party/vla_tools/dataset_inspector.py`**: Visual CLI inspector for dataset trajectory telemetry and joint bounds.

---

## 2. 1-Click Execution Guide

### Option 1: Run the 1-Click Third-Party Pipeline (Super Easy!)
```bash
bash /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/run_vla_pipeline.sh \
    pi0 \
    /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json
```

### Option 2: Inspect Dataset Telemetry
```bash
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/third_party/vla_tools/dataset_inspector.py \
    --dataset /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json
```

### Option 3: Fine-Tune VLA Models Manually
```bash
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/source/cobot/vla/train_vla.py \
    --model pi0 \
    --pretrained_hub lerobot/pi0_ur5 \
    --dataset /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/data/vla/cobot_vla_sample_dataset.json \
    --epochs 5
```

### Option 4: Closed-Loop Real-Time Inference
```bash
python3 /home/azeez/ws/dev_env/py_code/projects/viro_isaaclab/core/source/cobot/vla/inference_vla.py \
    --model pi0 \
    --prompt "reach and touch target red object"
```

---

## 3. Sub-Module File Organization

```
core/data/vla/                                # Master VLA Data Directory
├── cobot_vla_sample_dataset.json             # 3 sample trajectory data points for joint_1..joint_6
└── checkpoints/                              # Saved fine-tuned model weights (.pt)

third_party/vla_tools/                        # 1-Click Helper Tools Directory
├── dataset_inspector.py                      # CLI dataset breakdown & inspector
└── run_vla_pipeline.sh                       # 1-click dataset inspection -> fine-tune -> inference

core/source/cobot/vla/                        # VLA Model Implementation Directory
├── __init__.py                               # Python sub-package export
├── dataset_collector.py                      # Multi-modal trajectory demonstrator
├── pi0_model.py                              # Physical Intelligence pi0 / pi0.5 Flow Matching Policy
├── smol_vla_model.py                         # SmolVLA lightweight VLM Policy
├── act_model.py                              # Action Chunking with Transformers (ACT CVAE) Policy
├── train_vla.py                              # Pre-trained fine-tuning engine
├── inference_vla.py                          # Real-time closed-loop evaluation runner
└── README.md                                 # Master VLA guide (this file)
```
