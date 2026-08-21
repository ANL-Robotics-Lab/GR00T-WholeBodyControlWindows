# Convert RBY1 CSV motions to motion-library PKLs

The canonical converter is
`gear_sonic/data_process/rby1_retarget_seed_to_motionlib.py`. It converts an
already-retargeted RBY1 CSV into the PKL schema used by SONIC and
`rby1_sdk_replay.py`. It also has an explicitly heuristic G1-to-RBY1 projection
mode for plumbing tests.

Run the following commands from the `GR00T-WholeBodyControl` repository root.

## Install the standalone converter

Use a separate Python 3.11 environment so its NumPy 1.26 dependency does not
replace the NumPy 2 installation required by `rby1-sdk`:

```powershell
# First activate a Python 3.11 shell and confirm with: python --version
python -m venv .\rby1_hardware\.venv-converter
.\rby1_hardware\.venv-converter\Scripts\python.exe -m pip install -r `
  .\rby1_hardware\requirements-converter.txt
```

The module form used below makes the repository root importable without
installing the full `gear_sonic` training package.

## Reproduce the bundled dance PKL

The committed source is
`rby1_hardware/motions/source/guy_dancing_rby1_from_original_bvh_corrected.csv`.
Recreate the bundled output in a temporary file:

```powershell
.\rby1_hardware\.venv-converter\Scripts\python.exe -m `
  gear_sonic.data_process.rby1_retarget_seed_to_motionlib `
  --input .\rby1_hardware\motions\source\guy_dancing_rby1_from_original_bvh_corrected.csv `
  --output "$env:TEMP\guy_dancing_rby1_recreated.pkl" `
  --input-format rby1_csv `
  --input-joint-order mujoco `
  --fps-source 9 `
  --fps 9 `
  --rby1-root-mode wheel_feasible_planar_yaw_relative `
  --rby1-joint-smoothing-window 7 `
  --wheel-sign -1
```

This command retains all 116 source frames, interprets them at 9 fps to produce
a 12.778-second motion, smooths the joints with a seven-frame moving window, and
reconstructs a differential-drive-feasible root path. It reproduces this SHA-256:

```text
b9b85e2a948de35e93eea5da482e9ee4ac27547f21f18af0fd0a21de535f9e3f
```

Verify both the hash and the guarded replay trajectory:

```powershell
Get-FileHash "$env:TEMP\guy_dancing_rby1_recreated.pkl" -Algorithm SHA256

.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sdk_replay.py `
  "$env:TEMP\guy_dancing_rby1_recreated.pkl" `
  --target validate `
  --reference-mode absolute `
  --scale-absolute-pose `
  --fit-start-pose `
  --speed 0.10 `
  --torso-scale 0.10 `
  --arm-scale 0.75 `
  --head-scale 0.10
```

## Convert another RBY1 CSV

Use `--input-format rby1_csv` for an RBY1 trajectory containing all 24 joints.
The converter accepts either direct joint names such as `right_arm_0` or names
with a `_dof` suffix. Set `--input-joint-order` to the order actually written by
the producer. Do not guess this option: a wrong order produces a structurally
valid PKL with incorrect robot motion.

`--fps-source` describes the timing assigned to the input rows. `--fps` controls
the output frame rate. Setting both to the same lower value keeps every row and
slows the motion; setting a lower output FPS than source FPS downsamples by an
integer stride.

For the available options and schema help:

```powershell
.\rby1_hardware\.venv-converter\Scripts\python.exe -m `
  gear_sonic.data_process.rby1_retarget_seed_to_motionlib --help
```

## BVH inputs

This converter does not read BVH files. The high-fidelity path is:

```text
BVH -> modified GEM-X/SOMA retargeter -> RBY1 CSV -> this converter -> PKL
```

Keep the modified GEM-X pipeline in its own repository or provide a pinned
commit and setup instructions. The heuristic `bones_g1_csv_projection` mode is
useful for smoke testing a G1 CSV, but it is not a replacement for the RBY1 IK
retargeting stage.
