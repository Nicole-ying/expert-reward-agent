# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + gated_goal_reward | -10.16 | -10.16 | 0.00 | 1000.00 | contact_reward=0.237 gated_goal_reward=0.489 | new_best |
| 2 | contact_bonus + engine_penalty + gated_goal + height_reward | -63.81 | -10.16 | -53.64 | 1000.00 | contact_bonus=0.903 engine_penalty=-0.036 gated_goal=0.381 height_reward=0.680 | no_meaningful_improvement |
| 3 | contact_reward + engine_penalty + goal_reward + height_reward + landing_bonus | -20.70 | -10.16 | -10.54 | 1000.00 | contact_reward=0.250 engine_penalty=-0.074 goal_reward=0.441 height_reward=0.654 landing_bonus=0.913 | no_meaningful_improvement |
| 4 | contact_reward + descent_reward + engine_penalty + landing_bonus | -123.19 | -10.16 | -113.03 | 80.10 | contact_reward=0.016 descent_reward=-0.074 engine_penalty=-0.009 landing_bonus=0.038 | unsolved_stagnation_fresh_restart |
| 5 | global_speed_penalty + progress_reward + soft_landing | -104.19 | -10.16 | -94.02 | 68.55 | global_speed_penalty=-0.013 progress_reward=0.024 soft_landing=-0.066 | no_meaningful_improvement |
| 6 | engine_penalty + landing_reward + lateral_penalty + progress_reward | -122.77 | -10.16 | -112.61 | 68.30 | engine_penalty=-0.013 landing_reward=0.107 lateral_penalty=-0.029 progress_reward=0.014 | no_meaningful_improvement |
| 7 | engine_penalty + landing_reward + lateral_penalty + progress_reward | -122.03 | -10.16 | -111.87 | 68.30 | engine_penalty=-0.008 landing_reward=0.099 lateral_penalty=-0.029 progress_reward=0.033 | unsolved_stagnation_fresh_restart |
| 8 | approach_reward + soft_landing_penalty + upright_penalty | -5510.93 | -10.16 | -5500.77 | 586.85 | approach_reward=0.027 soft_landing_penalty=-0.007 upright_penalty=-0.004 | no_meaningful_improvement |
| 9 | approach_reward + soft_landing_penalty + upright_penalty | -5510.93 | -10.16 | -5500.77 | 586.85 | approach_reward=0.027 soft_landing_penalty=-0.007 upright_penalty=-0.004 | no_meaningful_improvement |
| 10 | contact_reward + engine_penalty + progress_reward + proximity_reward + upright_penalty | 59.18 | 59.18 | 0.00 | 764.05 | contact_reward=0.557 engine_penalty=-0.021 progress_reward=0.003 proximity_reward=0.400 upright_penalty=-0.001 | new_best |
