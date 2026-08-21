# Bundled RBY1 motions

`guy_dancing_rby1_trainable_9fps_smoothed7.pkl` is the verified 24-DOF RBY1
dance used by the SDK replay example. It contains 116 frames at 9 fps and uses
the RBY1 Model A / clean-MuJoCo joint order documented in
`gear_sonic/utils/rby1_order.py`.

Its reproducible input is committed at
`source/guy_dancing_rby1_from_original_bvh_corrected.csv`. The exact conversion
command and environment are documented in `../CSV_TO_PKL.md`; running that
command produces a byte-identical PKL.

SHA-256:
`b9b85e2a948de35e93eea5da482e9ee4ac27547f21f18af0fd0a21de535f9e3f`

The PKL is intentionally stored as a regular Git blob rather than Git LFS so a
normal GitHub clone includes an immediately usable example motion.
