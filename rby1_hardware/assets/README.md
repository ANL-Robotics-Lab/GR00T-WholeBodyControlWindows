# RBY1 replay assets

`model_v1.2.urdf` is the Rainbow Robotics RBY1 Model A v1.2 description used
by `rby1_sdk_replay.py` to read joint position and velocity limits. Keeping this
small file beside the runner removes the runtime dependency on the much larger
MoveIt workspace.

The replay runner only parses joint and `<limit>` elements. It does not load the
mesh paths referenced by the URDF, so the corresponding visual meshes are not
required for validation or SDK replay.

Source: `rby1-sdk/models/rby1a/urdf/model_v1.2.urdf` from the Rainbow Robotics
RBY1 SDK distribution.
