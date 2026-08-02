# Vision-Language-Action (VLA) Architecture & Execution Guide

This guide provides a first-principles, step-by-step walkthrough for running the entire Vision-Language-Action (VLA) pipeline.

Unlike standard RL (which maps proprioceptive state like joint angles to actions), a VLA model is a neural network (like a Transformer) that maps **RGB camera images** and **text prompts** directly to robot actions.

The VLA pipeline consists of 3 distinct phases:
1. **Data Collection**: Generate an `.h5` dataset of expert demonstrations (camera images + actions).
2. **Offline Training**: Fine-tune a pre-trained VLA model on your collected dataset.
3. **Closed-Loop Inference**: Deploy the trained model back into IsaacLab to execute actions live based on camera input.

---

## Prerequisites
Before running any scripts, ensure you are inside the IsaacLab docker container or your activated virtual environment, and change your directory to the workspace root:

```bash
# Example if using the native environment:
source isaac_sim51_lab/bin/activate
cd ~/IsaacLab  # Or wherever your workspace root is located
```
*Note: All paths in this guide assume you are executing from the workspace root.*

---

## Phase 1: Data Collection

To train a VLA, you first need data. The data collector spins up the IsaacLab simulator, runs a scripted or RL expert policy, and records the RGB camera feed and the expert's actions.

### How to Run
```bash
python core/source/cobot/vla/dataset_collector.py \
    --output core/data/vla/cobot_vla_dataset.h5 \
    --episodes 50
```

### Expected Input & Output
- **Input**: The script loads the `CobotEnvCfg_VLA` environment (which includes an RGB camera) and executes a scripted expert policy to reach and grasp.
- **Output**: Generates an HDF5 dataset file at `core/data/vla/cobot_vla_dataset.h5`.
- **Dataset Contents**: Each episode contains:
  - `image`: The RGB camera feed `[Time, Height, Width, 3]`.
  - `actions`: The target joint commands `[Time, 22]`.
  - `joint_pos` & `joint_vel`: Robot proprioception `[Time, 22]`.
  - `language_prompt`: The text instruction (e.g., "reach and touch the target red object").

---

## Phase 2: Offline Training (Fine-Tuning)

Once you have your `.h5` dataset, you need to train the model. Because training a VLA from scratch takes weeks, we download pre-trained weights (e.g., `pi0_ur5`) and **fine-tune** them on your dataset.

*Note: This phase does not require the IsaacLab simulator to be running.*

### How to Run
```bash
python core/source/cobot/vla/train_vla.py \
    --model pi0 \
    --pretrained_hub lerobot/pi0_ur5 \
    --dataset core/data/vla/cobot_vla_dataset.h5 \
    --epochs 10 \
    --output core/logs/vla/pi0_cobot_policy.pt
```

### Expected Input & Output
- **Input**: 
  - The `.h5` dataset generated in Phase 1.
  - Pre-trained base weights downloaded automatically from HuggingFace (`lerobot/pi0_ur5`).
- **Output**: 
  - A fine-tuned PyTorch checkpoint file saved to `core/logs/vla/pi0_cobot_policy.pt`. This file contains the updated neural network weights.

---

## Phase 3: Closed-Loop Inference

Now that the VLA model is fine-tuned, we can deploy it back into the simulation to test it. The simulator will feed live RGB images to the VLA, and the VLA will output joint actions.

### How to Run
```bash
python core/source/cobot/vla/inference_vla.py \
    --model pi0 \
    --ckpt core/logs/vla/pi0_cobot_policy.pt \
    --prompt "reach and touch target red object"
```

### Expected Input & Output
- **Input**: 
  - The trained `.pt` checkpoint from Phase 2.
  - Live RGB frames captured from the simulator's camera during execution.
  - The text `--prompt` defining the task.
- **Output**: 
  - The robot will attempt to execute the task autonomously using only the camera feed and the prompt.
  - Log output to the terminal showing the predicted joint actions at each step.

---

## Sub-Module File Organization

```text
core/source/cobot/vla/                        # VLA Model Implementation Directory
├── __init__.py                               
├── cobot_env_cfg.py                          # VLA environment config (spawns the camera)
├── dataset_collector.py                      # Phase 1: Collects images/actions into .h5
├── train_vla.py                              # Phase 2: Trains Neural Network on .h5 data
├── inference_vla.py                          # Phase 3: Evaluates policy in simulation
├── pi0_model.py                              # Model Arch: Physical Intelligence pi0
├── smol_vla_model.py                         # Model Arch: SmolVLA
├── act_model.py                              # Model Arch: Action Chunking Transformers
└── README.md                                 # This file
```
