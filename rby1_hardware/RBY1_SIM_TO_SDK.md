# RBY1 Isaac-state to SDK bridge

[`rby1_sim_to_sdk.py`](rby1_sim_to_sdk.py) mirrors a checkpoint that is still
running inside Isaac Lab. Isaac computes the policy observations, ten-frame
history, motion-reference features, policy output, mixed action decoding, and
simulated dynamics. The bridge reads the **realized simulator joint state** and
sends only the RBY1 Model A hardware channels through the same `rby1-sdk` command
builders and safety checks used by the PKL runner.

This is an open-loop policy bridge, not a sim-to-real feedback controller. The
policy observes its simulated RBY1, not the physical RBY1. SDK feedback is used
to stop on stale state, E-stop, loss of readiness, temperature, tracking error,
wheel error, or timing failure; it is not inserted into policy history.

## What is mapped

The imported Isaac articulation has 28 movable joints. The publisher selects
the following exact 24 and puts them in vendor Model A SDK order:

```text
right_wheel, left_wheel,
torso_0 ... torso_5,
right_arm_0 ... right_arm_6,
left_arm_0 ... left_arm_6,
head_0, head_1
```

`backwheel`, `backwheel2`, and the two finger joints are excluded. The two wheel
signs are inverted because the imported Isaac USD exposes +Y wheel axes while
the vendor Model A URDF/SDK uses -Y. Upper-body signs are unchanged. The SDK
receives 22 position-controlled upper-body joints and, only with
`--enable-wheels`, two velocity-controlled wheels.

The two processes use local UDP port `50070`. A strict packet repeats all 24
joint names, session ID, sequence number, simulator time, and 50 Hz rate. Isaac
waits at frame zero until the SDK receiver has checked and reached its initial
target. It then paces the episode to wall-clock time. An episode reset, missing
packet, sequence gap, or timing overrun stops the SDK stream.

## 1. Shadow test (no SDK connection)

Use two PowerShell terminals from the repository root. Start the receiver first:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sim_to_sdk.py `
  --target shadow `
  --listen 127.0.0.1:50070 `
  --verbose
```

In the Isaac Lab environment, start the saved checkpoint and publisher:

```powershell
conda activate env_lab
python .\rby1_hardware\rby1_checkpoint_eval.py `
  .\logs_rl\TRL_RBY1_Track\manager\universal_token\all_modes\RUN_NAME `
  --sdk-bridge 127.0.0.1:50070
```

Shadow mode does not import `rby1_sdk`, connect to a backend, power a joint, or
send a command. It validates mapping, packet timing, scaled URDF limits, and
writes `rby1_hardware/bridge_logs/sim_to_shadow_*.csv`. Complete this test before
starting either SDK simulator or hardware mode.

## 2. Vendor simulator test

Start the separately cloned `rby1-sim-isaac` SDK backend and D3D12 simulator as
for PKL replay. Then start this receiver before the checkpoint publisher:

docker run --rm -it   --name rby1-sdk-backend   --network host   -e DISPLAY="$DISPLAY"   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -v "$HOME/.Xauthority:/isaac-sim/.Xauthority:rw"   -u 1234:1234   --entrypoint /opt/rby1-sim-isaac/sdk/app/app_main   rainbowroboticsofficial/rby1-sim-isaac:0.10.7-a_v1.2   isaac

python .\src\simulation.py `
>>   --model a `
>>   --graphics-api d3d12 `
>>   --udp `
>>   --state-ip 127.0.0.1 `
>>   --state-port 5005 `
>>   --cmd-port 5006



```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sim_to_sdk.py `
  --target simulator `
  --address 127.0.0.1:50051 `
  --auto-enable `
  --reference-mode absolute `
  --torso-scale 0.10 `
  --arm-scale 1 `
  --head-scale 0.10 `
  --return-to-zero `
  --return-seconds 12 `
  --stationary-timeout 8 `
  --countdown 5 `
  --verbose
```

for just pkl repaly 
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
>>   .\rby1_hardware\rby1_sdk_replay.py `
>>   .\tmp_rby1_video_compare\guy_dancing_rby1_from_original_bvh_trainable_9fps_smoothed7.pkl `
>>   --target simulator `
>>   --address 127.0.0.1:50051 `
>>   --auto-enable `
>>   --reference-mode absolute `
>>   --fit-start-pose `
>>   --speed 0.10 `
>>   --torso-scale 0.1 `
>>   --arm-scale 0.75 `
>>   --head-scale 0.1 `
>>   --transition-seconds 12 `
>>   --stationary-timeout 8 `
>>   --countdown 5 `
>>   --verbose `
>>   --return-to-zero `
>>   --return-seconds 12

for checkpoint eval
C:\Users\bcarc_ziwaj0x\miniconda3\envs\env_lab\python.exe `
>>   .\rby1_hardware\rby1_checkpoint_eval.py `
>>   .\logs_rl\TRL_RBY1_Track\manager\universal_token\all_modes\sonic_rby1_action_scale_v2-20260715_092000\model_step_003150.pt `
>>   --motion .\rby1_hardware\motions\guy_dancing_rby1_trainable_9fps_smoothed7.pkl `
>>   --sdk-bridge 127.0.0.1:50070

Run the same `rby1_checkpoint_eval.py ... --sdk-bridge 127.0.0.1:50070`
command in the Isaac Lab terminal. Wheels are disabled. `relative` mode mirrors
the simulator's change from frame zero around the measured SDK start pose. This
avoids forcing the physical robot to equal the imported USD's absolute reset
pose. `absolute` mode is available for exact scaled SDK-coordinate poses, but
should first be validated against the vendor simulator.

Increase one scale at a time only after logs show clean tracking. Full
upper-body mirroring is:

```text
--torso-scale 1.0 --arm-scale 1.0 --head-scale 1.0
```

Do not enable wheels merely to make the command nominally 24-channel. Validate
upper-body behavior first. A later low-speed mobility test must explicitly add
`--enable-wheels --wheel-scale 0.10` in a cleared test area.

## 3. Guarded real-hardware trial

Use the same receiver with the robot's SDK address. The robot target requires
the exact confirmation token and at least a three-second countdown:

```powershell
.\rby1_hardware\.venv-sdk\Scripts\python.exe `
  .\rby1_hardware\rby1_sim_to_sdk.py `
  --target robot `
  --address ROBOT_IP:50051 `
  --reference-mode absolute `
  --torso-scale 0.10 `
  --arm-scale 0.25 `
  --head-scale 0.10 `
  --return-to-zero `
  --return-seconds 12 `
  --stationary-timeout 8 `
  --countdown 5 `
  --confirm-real-robot RUN_RBY1_MODEL_A `
  --verbose
  --soft-limit-policy warn `
  --confirm-relaxed-limits WARN_ONLY_RBY1_MODEL_A
```

Prepare power, servos, and the control manager manually, or add
`--auto-enable` only when that behavior is intended. Keep a trained operator at
the physical E-stop, begin with wheels disabled, clear the full arm workspace,
and use the reduced scales above. `--return-to-zero` runs only after Isaac marks
the episode as normally completed. An exception, Ctrl+C, publisher failure, or
watchdog stop holds measured position and zeros commanded wheel velocity instead
of initiating another trajectory.

Real hardware enforces projected position/dynamic and bridge timing limits by
default. For a deliberately warning-only diagnostic, add both explicit options:

```powershell
--soft-limit-policy warn `
--confirm-relaxed-limits WARN_ONLY_RBY1_MODEL_A
```

This profile clamps projected joint positions at the vendor URDF bounds, warns
instead of aborting on the post-limiter dynamic envelope, holds the latest
upper-body target with zero wheel velocity across source-packet timeouts, and
rebases accumulated bridge lag. `--soft-limit-policy enforce` restores the
default abort behavior. Warning-only mode does **not** relax loss of SDK robot
state, E-stop/control-manager readiness, temperature, measured tracking or
wheel error, packet session/order, or command-transport failures; those remain
hard stops. Keep an operator at the physical E-stop throughout this diagnostic.

## Process order and shutdown

1. Start the bridge receiver; it waits for an Isaac hello.
2. Start checkpoint evaluation with `--sdk-bridge`.
3. Isaac freezes at reset while the receiver performs SDK checks and any
   initial transition.
4. The receiver acknowledges readiness and refreshes the settled hold command
   while Isaac computes its first policy/physics state. This keeps the vendor
   SDK command-stream lease alive without advancing the motion.
   `--first-packet-timeout` allows 5 seconds for this one-time CUDA warm-up by
   default.
5. The first state starts the shared real-time clock. Local shadow and vendor
   simulator tests allow up to 5 seconds for an active packet because Windows
   GUI/CUDA initialization can span several policy frames. If accumulated lag
   exceeds 1 second, both local pacing clocks rebase instead of repeatedly
   charging startup/rendering delay to later frames. The bridge refreshes the
   latest accepted hold during each gap. `--target robot` retains the stricter
   0.2-second packet and 0.1-second lag defaults and never rebases late timing.
   Local shadow/vendor-simulator tests use the physical URDF position bounds
   without an additional inward margin and clamp an incompatible checkpoint
   pose at those bounds. Every newly clamped joint is reported, because that
   pose cannot be reproduced exactly. Real-hardware mode retains a 0.01-rad
   margin and rejects out-of-range targets; clamp mode is prohibited there.
6. Projected poses pass through a software rate limiter using the same scaled
   URDF velocity and acceleration limits sent to the SDK. Fast Isaac targets
   are followed with bounded lag; raw positions outside the safe joint margins
   are still rejected immediately. The CSV's `rate_limiter_lag` column records
   the maximum instantaneous difference from the projected simulator target.
7. Normal episode completion stops mobility and optionally returns to zero.

Both sides default to loopback transport. Remote UDP must be explicitly enabled
on both commands and should only be used on an isolated trusted robot network.
The bridge protocol is not authenticated or encrypted.
