# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_velocity_reward + stability_penalty | -123.55 | -123.55 | 0.00 | 206.85 | forward_velocity_reward=0.234 stability_penalty=-0.240 | new_best |
| 2 | forward_velocity_reward + orientation_reward + stability_penalty | -50.90 | -50.90 | 0.00 | 407.80 | forward_velocity_reward=0.316 orientation_reward=0.092 stability_penalty=-0.247 | new_best |
| 3 | forward_velocity_reward + orientation_reward + stability_penalty | -137.45 | -50.90 | -86.55 | 309.10 | forward_velocity_reward=0.318 orientation_reward=0.043 stability_penalty=-0.286 | no_meaningful_improvement |
| 4 | forward_velocity_reward + orientation_reward + stability_penalty | -49.53 | -49.53 | 0.00 | 819.40 | forward_velocity_reward=0.342 orientation_reward=0.164 stability_penalty=-0.219 | same_skeleton_persistent_negative_fresh_restart |
| 5 | angular_velocity_penalty + forward_velocity_reward + lateral_velocity_penalty + orientation_reward | 11.05 | 11.05 | 0.00 | 252.35 | angular_velocity_penalty=-0.113 forward_velocity_reward=0.448 lateral_velocity_penalty=-0.106 orientation_reward=0.073 | new_best |
| 6 | angular_velocity_penalty + forward_velocity_reward + height_deviation_penalty + lateral_velocity_penalty + orientation_reward | 12.31 | 12.31 | 0.00 | 161.30 | angular_velocity_penalty=-0.129 forward_velocity_reward=0.416 height_deviation_penalty=-0.385 lateral_velocity_penalty=-0.119 orientation_reward=0.031 | new_best |
| 7 | angular_velocity_penalty + forward_velocity_reward + height_boundary_penalty + lateral_velocity_penalty + orientation_reward | -131.92 | 12.31 | -144.24 | 376.25 | angular_velocity_penalty=-0.112 forward_velocity_reward=0.388 height_boundary_penalty=-0.033 lateral_velocity_penalty=-0.100 orientation_reward=0.026 | unsolved_stagnation_fresh_restart |
| 8 | contact_reward + forward_encouragement + height_management + lateral_penalty + stability_cost | -19.20 | 12.31 | -31.51 | 10.10 | contact_reward=0.000 forward_encouragement=0.007 height_management=-0.632 lateral_penalty=-0.346 stability_cost=-3.246 | no_meaningful_improvement |
| 10 | forward_tracking + stability_cost | -17.31 | 12.31 | -29.63 | 9.30 | forward_tracking=-13.716 stability_cost=-1.704 | unsolved_stagnation_fresh_restart |
