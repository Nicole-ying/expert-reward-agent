# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.49 | -159.49 | 0.00 | 303.35 | contact_reward=0.000 orientation_penalty=-0.008 progress_reward=-0.002 vel_damping=-0.004 | new_best |
| 2 | contact_reward + orientation_penalty + progress_reward + vel_damping | -168.02 | -159.49 | -8.53 | 387.20 | contact_reward=0.000 orientation_penalty=-0.007 progress_reward=-0.002 vel_damping=-0.008 | no_meaningful_improvement |
| 3 | contact_reward + orientation_penalty + progress_reward + vel_damping | -124.55 | -124.55 | 0.00 | 592.80 | contact_reward=54.239 orientation_penalty=-0.015 progress_reward=0.002 vel_damping=-0.113 | new_best |
| 4 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.20 | -124.55 | -34.65 | 1000.00 | contact_reward=12.398 orientation_penalty=-0.014 progress_reward=0.001 vel_damping=-0.091 | same_skeleton_persistent_negative_fresh_restart |
| 5 | action_penalty + landing_reward + progress_reward | -121.58 | -121.58 | 0.00 | 68.35 | action_penalty=-0.003 landing_reward=0.025 progress_reward=0.006 | new_best |
| 6 | action_penalty + landing_reward + progress_reward | -103.79 | -103.79 | 0.00 | 1000.00 | action_penalty=-0.008 landing_reward=5.456 progress_reward=0.001 | new_best |
| 7 | action_penalty + contact_reward + descent_reward + progress_reward | -120.10 | -103.79 | -16.30 | 76.90 | action_penalty=-0.005 contact_reward=0.516 descent_reward=0.041 progress_reward=0.016 | no_meaningful_improvement |
| 8 | contact_quality + landing_gate_penalty + progress | -477.40 | -103.79 | -373.61 | 112.90 | contact_quality=0.000 landing_gate_penalty=-0.031 progress=0.014 | no_meaningful_improvement |
| 9 | contact_quality + landing_gate_penalty + progress | -477.40 | -103.79 | -373.61 | 112.90 | contact_quality=0.000 landing_gate_penalty=-0.031 progress=0.014 | unsolved_stagnation_fresh_restart |
| 10 | attitude_penalty + contact_bonus + proximity_penalty + vel_penalty | -68.14 | -68.14 | 0.00 | 70.80 | attitude_penalty=-0.010 contact_bonus=0.003 proximity_penalty=-0.984 vel_penalty=-0.081 | new_best |
