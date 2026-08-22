# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_penalty + posture_penalty + progress_reward + vertical_penalty | -61.55 | -61.55 | 0.00 | 253.05 | angular_penalty=-0.000 posture_penalty=-0.001 progress_reward=0.069 vertical_penalty=-0.001 | new_best |
| 2 | angular_penalty + posture_gate + progress_reward + vertical_penalty | -61.57 | -61.55 | -0.02 | 303.45 | angular_penalty=-0.000 posture_gate=0.649 progress_reward=0.046 vertical_penalty=-0.000 | no_meaningful_improvement |
| 3 | air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | -59.20 | -59.20 | 0.00 | 394.20 | air_penalty=-0.000 angular_penalty=-0.000 posture_gate=0.664 progress_reward=0.043 vertical_penalty=-0.000 | new_best |
| 4 | air_penalty + angular_penalty + posture_gate + progress_reward + vertical_penalty | -65.67 | -59.20 | -6.47 | 148.40 | air_penalty=-0.000 angular_penalty=-0.000 posture_gate=0.599 progress_reward=0.047 vertical_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 5 | action_cost + ang_vel_penalty + posture_penalty + progress_reward | -65.16 | -59.20 | -5.96 | 190.30 | action_cost=-0.019 ang_vel_penalty=-0.000 posture_penalty=-0.057 progress_reward=0.382 | no_meaningful_improvement |
| 6 | action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward | -52.73 | -52.73 | 0.00 | 323.50 | action_cost=-0.019 air_penalty=-0.140 ang_vel_penalty=-0.000 posture_penalty=-0.054 progress_reward=0.434 | new_best |
| 7 | action_cost + air_penalty + ang_vel_penalty + posture_penalty + progress_reward + vertical_speed_penalty | -52.73 | -52.73 | 0.00 | 323.50 | action_cost=-0.019 air_penalty=-0.140 ang_vel_penalty=-0.000 posture_penalty=-0.054 progress_reward=0.434 | unsolved_stagnation_fresh_restart |
| 9 | progress_gated + vertical_penalty | -29.45 | -29.45 | 0.00 | 372.70 | progress_gated=0.200 vertical_penalty=-0.000 | new_best |
| 10 | air_penalty + progress_gated | -66.62 | -29.45 | -37.17 | 167.75 | air_penalty=-0.075 progress_gated=0.178 | no_meaningful_improvement |
