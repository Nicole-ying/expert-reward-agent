# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_progress + lateral_drift_hinge + roll_pitch_hinge | -336.84 | -336.84 | 0.00 | 944.40 | forward_progress=0.558 lateral_drift_hinge=-0.073 roll_pitch_hinge=-0.081 | new_best |
| 2 | forward_progress + lateral_drift_hinge + roll_pitch_hinge | -75.45 | -75.45 | 0.00 | 694.40 | forward_progress=0.592 lateral_drift_hinge=-0.102 roll_pitch_hinge=-0.086 | new_best |
| 3 | forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge | -186.37 | -75.45 | -110.92 | 752.35 | forward_progress=0.595 height_boundary_hinge=-0.008 lateral_drift_hinge=-0.089 roll_pitch_hinge=-0.084 | no_meaningful_improvement |
| 4 | descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge | 5.50 | 5.50 | 0.00 | 968.50 | descent_early_penalty=-0.001 forward_progress=0.694 lateral_drift_hinge=-0.086 roll_pitch_hinge=-0.085 | new_best |
| 5 | action_cost + angular_factor + forward_reward + gate + height_factor + lateral_factor | -345.87 | 5.50 | -351.38 | 306.40 | action_cost=0.004 angular_factor=0.392 forward_reward=0.189 gate=0.213 height_factor=0.663 | unsolved_stagnation_fresh_restart |
| 6 | forward_progress + survival_bonus | -585.81 | 5.50 | -591.32 | 663.70 | forward_progress=0.160 survival_bonus=0.500 | no_meaningful_improvement |
| 7 | forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty | -415.21 | 5.50 | -420.72 | 299.05 | forward_reward=0.260 height_high_penalty=-0.002 height_low_penalty=-0.026 lateral_penalty=-0.099 pitch_penalty=-0.092 | no_meaningful_improvement |
| 8 | forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty | -845.86 | 5.50 | -851.36 | 796.90 | forward_reward=0.324 height_high_penalty=-0.001 height_low_penalty=-0.020 lateral_penalty=-0.090 pitch_penalty=-0.081 | unsolved_stagnation_fresh_restart |
| 9 | action_penalty + ascent_penalty + descent_penalty + forward_reward + joint_vel_penalty + lateral_penalty | -375.55 | 5.50 | -381.05 | 662.05 | action_penalty=-0.005 ascent_penalty=-0.001 descent_penalty=-0.003 forward_reward=2.216 joint_vel_penalty=-0.065 | no_meaningful_improvement |
| 10 | action_penalty + forward_reward + height_gate + joint_vel_penalty + lateral_penalty + pitch_vel_penalty | -1313.79 | 5.50 | -1319.29 | 757.45 | action_penalty=-0.003 forward_reward=0.187 height_gate=0.644 joint_vel_penalty=-0.018 lateral_penalty=-0.008 | no_meaningful_improvement |
