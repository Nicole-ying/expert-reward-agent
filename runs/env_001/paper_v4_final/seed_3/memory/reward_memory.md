# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + lateral_drift_penalty + progress + stability_penalty | -111.37 | -111.37 | 0.00 | 101.60 | landing_bonus=0.057 lateral_drift_penalty=-0.033 progress=0.147 stability_penalty=-0.028 | new_best |
| 2 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -103.62 | -103.62 | 0.00 | 68.70 | angvel_penalty=-0.020 landing_bonus=0.016 lateral_drift_penalty=-0.039 progress_gated=0.144 | new_best |
| 3 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -71.40 | -71.40 | 0.00 | 70.55 | angvel_penalty=-0.019 landing_bonus=0.017 lateral_drift_penalty=-0.039 progress_gated=0.146 | new_best |
| 4 | angvel_penalty + landing_reward + lateral_pos_penalty + progress_gated | -178.95 | -71.40 | -107.55 | 915.45 | angvel_penalty=-0.001 landing_reward=1.111 lateral_pos_penalty=-0.025 progress_gated=0.003 | no_meaningful_improvement |
| 5 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 30.06 | 30.06 | 0.00 | 668.35 | angvel_penalty=-0.002 contact_landing_reward=2.151 lateral_pos_penalty=-0.017 progress_gated=0.026 | new_best |
| 6 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -13.38 | 30.06 | -43.44 | 936.95 | angvel_penalty=-0.002 contact_landing_reward=2.420 lateral_pos_penalty=-0.046 progress_gated=0.108 | no_meaningful_improvement |
| 7 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 123.31 | 123.31 | 0.00 | 1000.00 | angvel_penalty=-0.002 contact_landing_reward=2.166 lateral_pos_penalty=-0.008 progress_gated=0.120 | new_best |
| 8 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -73.51 | 123.31 | -196.82 | 69.80 | angvel_penalty=-0.009 contact_landing_reward=0.048 lateral_pos_penalty=-0.003 progress_gated=0.459 | no_meaningful_improvement |
| 9 | angle_term_penalty + angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -24.78 | 123.31 | -148.09 | 143.15 | angle_term_penalty=-0.002 angvel_penalty=-0.007 contact_landing_reward=0.051 lateral_pos_penalty=-0.003 progress_gated=0.442 | no_meaningful_improvement |
| 10 | angle_penalty + angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -74.63 | 123.31 | -197.93 | 69.90 | angle_penalty=-0.003 angvel_penalty=-0.010 contact_landing_reward=0.048 lateral_pos_penalty=-0.003 progress_gated=0.459 | unsolved_high_achievement_continue_from_best |
