# Guarded RBY1 SDK motion replay

[`rby1_sdk_replay.py`](rby1_sdk_replay.py) loads the 24-DOF motion-library PKLs
produced by `rby1_retarget_seed_to_motionlib.py`. It uses the same Model A joint
order as the vendor SDK. Torso, arms, and head are streamed as position commands;
optional wheel velocities are derived from the two wheel-angle columns.

Run it in a Python environment where `numpy`, `joblib`, and the vendor
`rby1_sdk` package are importable. The tested Windows versions are pinned in
[`requirements-sdk.txt`](requirements-sdk.txt). Offline validation only needs
`numpy` and `joblib`. Root translation/orientation and gripper data are not
commanded, and the runner does not perform geometric self-collision or
environment-collision planning.

The runner defaults to offline validation. It does not import the SDK, connect to
a server, or send commands in this mode:

```powershell
python rby1_hardware/rby1_sdk_replay.py `
  rby1_hardware/motions/guy_dancing_rby1_trainable_9fps_smoothed7.pkl `
  --reference-mode absolute --scale-absolute-pose --fit-start-pose `
  --speed 0.10 --torso-scale 0.10 --arm-scale 0.75 --head-scale 0.10
```

The original full-amplitude PKL is always checked for diagnostics. A second check
validates the trajectory after the selected speed, amplitude, reference mode, and
wheel settings have been applied. In relative mode, offline validation assumes a
zero starting pose; execution repeats the check around the measured starting pose.

## Vendor simulator

Clone the separately maintained simulator fork at the repository root, then
start the Model A v1.2 SDK backend and Windows D3D12 simulator by following that
fork's README:

```bash
git clone https://github.com/ANL-Robotics-Lab/rby1-sim-isaac.git
git -C rby1-sim-isaac checkout 8e7e6fa78a117e4d85e39953451f7c635ccc5a3d
```

Then run the bundled, reduced-amplitude replay. This explicitly allows the script
to prepare the simulated robot, but wheels remain disabled:

```powershell
python rby1_hardware/rby1_sdk_replay.py `
  rby1_hardware/motions/guy_dancing_rby1_trainable_9fps_smoothed7.pkl `
  --target simulator --address 127.0.0.1:50051 --auto-enable `
  --reference-mode absolute --scale-absolute-pose --fit-start-pose `
  --speed 0.10 --torso-scale 0.10 --arm-scale 0.75 --head-scale 0.10 `
  --transition-seconds 12 --return-to-zero --return-seconds 12 `
  --stationary-timeout 8 --countdown 5 --verbose
```

The bundled first trial uses absolute source coordinates but scales the complete
pose, including frame zero, toward zero to lower self-collision risk. After
verifying joint directions, tracking, and E-stop behavior in the simulator,
exact source amplitude can be requested explicitly:

```powershell
python rby1_hardware/rby1_sdk_replay.py MOTION.pkl `
  --target simulator --auto-enable --reference-mode absolute `
  --speed 1 --torso-scale 1 --arm-scale 1 --head-scale 1
```

That command will still refuse to execute if the resulting path violates the Model
A v1.2 URDF limits. Do not bypass those failures by clipping individual frames;
repair/re-retarget the source or use the relative reduced-amplitude trial.

Wheels require a separate opt-in:

```powershell
python rby1_hardware/rby1_sdk_replay.py MOTION.pkl `
  --target simulator --auto-enable --enable-wheels --wheel-scale 0.10
```

Confirm both wheel directions at very low scale before increasing it. The
IsaacLab-specific wheel sign conversion is deliberately not applied here because
the PKL and vendor Model A SDK share the clean model convention.

## Optional all-zero finish

Add `--return-to-zero` to return every torso, arm, and head joint to `0 rad`
after all selected PKL frames finish normally. `--return-seconds` sets the
minimum duration; when omitted, it uses `--transition-seconds`. The runner
automatically lengthens the minimum-jerk transition if required by the scaled
URDF velocity or acceleration limits, settles the final motion target before
moving, and verifies convergence at zero.

RBY1's wheels are velocity-controlled. The option commands zero wheel velocity
but does not unwind wheel angles or drive the base back to its starting location.
The return is deliberately skipped after cancellation, an E-stop, stale state,
tracking failure, or any other abnormal termination; those paths hold the
measured pose instead of initiating additional motion.

The zero return is joint-space interpolation, not collision-planned motion.
Verify the complete return in the simulator at reduced amplitude before using
it on physical hardware, and keep the E-stop available throughout the return.

## Physical robot

Before connecting, keep a trained operator at the physical E-stop, clear the work
area, verify Model A v1.2 joint limits with Rainbow Robotics, and prepare the robot
through the normal hardware procedure. First run state-relative, wheels-off motion:

```powershell
python rby1_hardware/rby1_sdk_replay.py MOTION.pkl `
  --target robot --address ROBOT_IP:50051 `
  --confirm-real-robot RUN_RBY1_MODEL_A --end-frame 20
```

`--end-frame 20` intentionally limits the first hardware trial. Remove it only
after inspecting the resulting CSV and replaying the full path in the vendor
simulator.

Without `--auto-enable`, the runner refuses to power, servo, reset faults, or enable
the control manager. If the normal hardware procedure calls for the SDK process to
do the first three operations, add `--auto-enable`. Fault reset remains a separate
explicit `--reset-faults` option and should only be used after inspecting the fault.

Every execution is revalidated against measured joint positions and monitors state
freshness, joint readiness, E-stop state, temperature, control-manager state,
tracking error, and command-loop lag. On exit it sends zero wheel velocity (when it
owned mobility), holds the measured upper-body pose, cancels the stream, and writes
a timestamped state/command CSV under `rby1_hardware/replay_logs`. Logs now include
a `phase` column and record `return_to_zero` when that phase completes.

After an aborted simulator replay, reset Isaac Sim to its zero pose with Backspace
before running again with `--fit-start-pose`. The runner deliberately rejects a
nonzero fitted simulator start so an old abort pose cannot silently shift the next
trajectory. If the vendor SDK backend reports joint tracking faults, reduce
`--speed` or motion amplitude rather than raising the local tracking watchdog: the
backend enforces its own tighter tracking threshold independently.
