# RBY1 SDK motion replay

This directory is the self-contained client package for validating and replaying
24-DOF RBY1 motion-library PKLs. It includes the guarded SDK runner, the Model A
v1.2 URDF used for safety limits, and a reproducible dance example. Run all
commands below from the `GR00T-WholeBodyControl` repository root.

The Isaac Sim application is maintained separately. Clone the tested Windows SDK
integration fork as a top-level directory beside `rby1_hardware`:

```powershell
git clone https://github.com/ANL-Robotics-Lab/rby1-sim-isaac.git `
  .\rby1-sim-isaac
git -C .\rby1-sim-isaac checkout `
  8e7e6fa78a117e4d85e39953451f7c635ccc5a3d
```

That simulator checkout is deliberately ignored by this repository. Its patched
README documents the WSL2 SDK backend and native Windows D3D12 simulator startup.

## Install the isolated Windows SDK client

RBY1 SDK 0.9.1 uses NumPy 2, while the native Isaac Sim and CSV conversion
environments use NumPy 1.26. Keep them isolated:

```powershell
# First activate a Python 3.11 shell and confirm with: python --version
python -m venv .\rby1_hardware\.venv-sdk
.\rby1_hardware\.venv-sdk\Scripts\python.exe -m pip install -r `
  .\rby1_hardware\requirements-sdk.txt
```

Confirm that the bundled motion and reduced first-trial trajectory are valid:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sdk_replay.py `
  .\rby1_hardware\motions\guy_dancing_rby1_trainable_9fps_smoothed7.pkl `
  --target validate `
  --reference-mode absolute `
  --scale-absolute-pose `
  --fit-start-pose `
  --speed 0.10 `
  --torso-scale 0.10 `
  --arm-scale 0.75 `
  --head-scale 0.10 `
  --return-to-zero `
  --return-seconds 12
```

## First simulator replay

Start the WSL SDK backend and native Windows D3D12 simulator from the separate
`rby1-sim-isaac` checkout. Reset Isaac Sim to its zero pose before each fitted
run. In a third PowerShell terminal, run:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sdk_replay.py `
  .\rby1_hardware\motions\guy_dancing_rby1_trainable_9fps_smoothed7.pkl `
  --target simulator `
  --address 127.0.0.1:50051 `
  --auto-enable `
  --reference-mode absolute `
  --scale-absolute-pose `
  --fit-start-pose `
  --speed 0.10 `
  --torso-scale 0.10 `
  --arm-scale 0.75 `
  --head-scale 0.10 `
  --transition-seconds 12 `
  --return-to-zero `
  --return-seconds 12 `
  --stationary-timeout 8 `
  --countdown 5 `
  --verbose
```

`--scale-absolute-pose` makes the reduced scales apply to the initial pose as
well as subsequent movement. Wheels remain disabled for this first
collision-conscious trial. Keep the physical E-stop available when progressing
to real hardware. After every selected PKL frame completes normally,
`--return-to-zero` stops the wheels, settles the final target, and streams the
torso, arms, and head back to `0 rad`. It does not drive the mobile base back to
its original location.

## Convert an RBY1 CSV

The example's corrected source CSV and converter are included. Conversion uses a
second lightweight environment and does not require Isaac Sim or `rby1-sdk`.
See [`CSV_TO_PKL.md`](CSV_TO_PKL.md) for the exact hash-reproducing command,
accepted CSV schemas, FPS behavior, and the separate BVH-to-CSV step.

See [`RBY1_SDK_REPLAY.md`](RBY1_SDK_REPLAY.md) for validation, wheel, logging,
and guarded real-hardware instructions.

## Try a trained checkpoint

Use [`rby1_checkpoint_eval.py`](rby1_checkpoint_eval.py) from the Isaac Lab
training environment to run a saved RBY1 checkpoint with its exact training
observations and configuration:

```powershell
conda activate env_lab
python .\rby1_hardware\rby1_checkpoint_eval.py `
  .\logs_rl\TRL_RBY1_Track\manager\universal_token\all_modes\RUN_NAME
```

It opens the Isaac Lab viewer and exits after one episode. It accepts either an
experiment directory or a checkpoint file, repairs the old ignored dance-motion
path and renamed filename using the bundled PKL, and supports `--dry-run`, `--motion`, `--loop`, and
`--export-onnx`. See [`RBY1_CHECKPOINT_EVAL.md`](RBY1_CHECKPOINT_EVAL.md).

By default the checkpoint launcher is simulator-only. Its explicit
`--sdk-bridge 127.0.0.1:50070` mode can instead publish the simulator's realized
24-joint state to [`rby1_sim_to_sdk.py`](rby1_sim_to_sdk.py). The SDK receiver
reuses the PKL runner's guarded command path and defaults to non-commanding
shadow mode. See [`RBY1_SIM_TO_SDK.md`](RBY1_SIM_TO_SDK.md) and complete the
shadow and vendor-simulator stages before considering real hardware.
