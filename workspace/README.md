# Workspace — Persistent Data

This folder stores all runtime output from training and evaluation.
It is **mounted into the Docker container** at `/workspace/` and persists
across container restarts.

## Folder layout

```
workspace/
├── logs/                 ← TensorBoard logs, training metrics
│   ├── rsl_rl/           ← RSL-RL TensorBoard event files
│   │   └── <task_id>/
│   │       ├── events.out.tfevents.*   ← TensorBoard data
│   │       └── checkpoints/            ← RSL-RL checkpoints (model_*.pt)
│   ├── eval/             ← Headless behavior reports (.html)
│   └── <other>/          ← Custom log dirs
│
├── models/               ← Periodic model checkpoints (your .pt files)
│   └── <task_id>/        ← One folder per registered task
│       ├── model_50.pt
│       ├── model_100.pt
│       └── model_latest.pt
│
└── data/                 ← Videos, datasets, exported policies
    └── <task_id>/
        ├── videos/       ← MP4 clips from training/play
        └── exported/     ← ONNX / TorchScript policy exports
```

## Key variables

### `logs/` (TensorBoard logs)
- Automatically created by RSL-RL when you run `train.py`.
- Each task gets its own subdirectory: `logs/rsl_rl/<task_id>/`.
- Open in browser: `http://localhost:6006` (auto-started in viz mode).

### `models/` (Checkpoints)
- **You should create this folder** before training if it doesn't exist.
- Checkpoints are saved at intervals defined by `save_interval` in
  `agents/rsl_rl_ppo_cfg.py`.
- Naming convention: `model_<iteration>.pt` (e.g., `model_500.pt`).
- A `model_latest.pt` symlink is always updated to the most recent.
- Resume training:
  ```bash
  python rl/train.py --task Isaac-Template-Walk-v0 \
      --resume --checkpoint /workspace/models/Isaac-Template-Walk-v0/model_500.pt
  ```

### `data/` (Videos & exports)
- Videos are saved when you pass `--video` to train.py or play.py.
- Exported policies (ONNX, TorchScript) go in `exported/`.
