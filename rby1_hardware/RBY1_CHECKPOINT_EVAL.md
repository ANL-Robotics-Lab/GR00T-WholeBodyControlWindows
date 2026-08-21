# RBY1 checkpoint evaluation

[`rby1_checkpoint_eval.py`](rby1_checkpoint_eval.py) is a small, Windows-friendly
launcher for testing an RBY1 training checkpoint in the original Isaac Lab
environment. It finds the checkpoint's saved `config.yaml`, uses the same policy
and observation construction as training, disables observation corruption, and
exits after one episode by default.

This is intentionally separate from [`rby1_sdk_replay.py`](rby1_sdk_replay.py).
The SDK replay runner commands known joint trajectories. The learned checkpoint
also consumes gravity and base angular velocity, ten frames of proprioceptive and
action history, and future motion-reference features. Those inputs are currently
computed inside Isaac Lab. The launcher itself never imports or connects to the
vendor SDK.

## Interactive checkpoint trial

Activate the same Isaac Lab environment used for training, then run this from the
repository root. A checkpoint file or its experiment directory is accepted:

```powershell
conda activate env_lab
python .\rby1_hardware\rby1_checkpoint_eval.py `
  .\logs_rl\TRL_RBY1_Track\manager\universal_token\all_modes\RUN_NAME
```

The default `g1` encoder follows the complete motion-library reference. The
launcher also disables policy/tokenizer noise and adds `run_once=true`, which
avoids leaving the evaluator cycling indefinitely. Add `--loop` only when an
indefinite viewer session is wanted.

Old training configs in this workspace refer to the ignored
`tmp_rby1_video_compare` directory. If that path is absent, the launcher detects
the matching motion under `rby1_hardware/motions` (including the bundled demo's
shortened `from_original_bvh` filename) and uses the bundled dance PKL.
For a different motion or checkpoint, pass the matching source explicitly:

```powershell
python .\rby1_hardware\rby1_checkpoint_eval.py .\path\to\last.pt `
  --motion .\path\to\matching_motion.pkl
```

Use `--dry-run` to validate all paths and print the underlying
`eval_agent_trl.py` command without starting Isaac Sim. If the launcher itself is
started from another environment, select the Isaac Lab interpreter explicitly:

```powershell
python .\rby1_hardware\rby1_checkpoint_eval.py .\path\to\last.pt `
  --python C:\Users\USER\miniconda3\envs\env_lab\python.exe `
  --dry-run
```

Other useful options are `--headless`, `--num-envs N`, `--max-steps N`, and a
repeatable `--override HYDRA_KEY=VALUE`. Run `--help` for the full interface.

## Publish the running simulator to the SDK bridge

The optional `--sdk-bridge HOST:PORT` path leaves all observations, history,
reference construction, and policy inference in Isaac. It selects the realized
24-joint simulator state after action decoding and publishes that to a separate
SDK process:

```powershell
python .\rby1_hardware\rby1_checkpoint_eval.py .\path\to\last.pt `
  --sdk-bridge 127.0.0.1:50070
```

This mode forces one environment and one episode, freezes Isaac at frame zero
until the receiver is ready, and wall-clock paces the evaluation. It rejects
`--loop` because an automatic simulator reset would create a discontinuous
hardware target. Start with the non-commanding receiver and follow the staged
instructions in [`RBY1_SIM_TO_SDK.md`](RBY1_SIM_TO_SDK.md).

## Export the deployment artifacts

The same launcher can invoke the repository's checkpoint-configured ONNX export:

```powershell
python .\rby1_hardware\rby1_checkpoint_eval.py .\path\to\last.pt `
  --export-onnx
```

The exported models are written below the checkpoint directory's `exported`
folder, with `model_config.yaml` beside the checkpoint. ONNX remains necessary
for a future standalone policy runner that reconstructs observations from real
hardware. It is not needed by the simulator-state bridge because that bridge
deliberately keeps inference and history inside Isaac Lab.
