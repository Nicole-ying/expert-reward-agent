# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | attitude_penalty + proximity_progress + soft_landing_bonus | -2.98 | -2.98 | 0.00 | 813.35 | attitude_penalty=-0.017 proximity_progress=0.257 soft_landing_bonus=1.013 | new_best |
| 2 | approach + attitude_penalty + contact_reward + fuel_cost + hard_landing + position_quality | -113.74 | -2.98 | -110.76 | 68.40 | approach=0.121 attitude_penalty=-0.129 contact_reward=0.058 fuel_cost=-0.020 hard_landing=-0.065 | no_meaningful_improvement |
| 3 | altitude_reward + attitude_penalty + ground_speed_penalty + partial_contact + proximity_progress + soft_landing | -61.36 | -2.98 | -58.38 | 1000.00 | altitude_reward=0.063 attitude_penalty=-0.039 ground_speed_penalty=-0.154 partial_contact=0.690 proximity_progress=0.114 | no_meaningful_improvement |
| 4 | attitude_penalty + descent_bonus + fuel_cost + proximity_progress + soft_landing | -114.32 | -2.98 | -111.34 | 68.35 | attitude_penalty=-0.031 descent_bonus=0.104 fuel_cost=-0.011 proximity_progress=0.484 soft_landing=0.067 | unsolved_stagnation_fresh_restart |
| 5 | attitude_penalty + progress_reward | 146.04 | 146.04 | 0.00 | 963.85 | attitude_penalty=-0.009 progress_reward=0.051 | new_best |
| 6 | attitude_penalty + contact_reward + crash_penalty + fuel_cost + landing_bonus + progress_reward | -53.29 | 146.04 | -199.33 | 1000.00 | attitude_penalty=-0.018 contact_reward=0.515 crash_penalty=-0.007 fuel_cost=-0.028 landing_bonus=1.257 | no_meaningful_improvement |
| 7 | attitude_penalty + crash_penalty + descent_bonus + progress_reward + soft_landing | 59.58 | 146.04 | -86.46 | 417.60 | attitude_penalty=-0.022 crash_penalty=-0.000 descent_bonus=0.064 progress_reward=0.283 soft_landing=2.266 | no_meaningful_improvement |
| 8 | attitude_penalty + descent_bonus + landing_bonus + progress_reward + stability_reward | -121.74 | 146.04 | -267.78 | 68.35 | attitude_penalty=-0.033 descent_bonus=0.104 landing_bonus=0.084 progress_reward=0.482 stability_reward=0.001 | unsolved_high_achievement_continue_from_best |
| 9 | attitude_penalty + crash_penalty + landing_bonus + progress_reward | 107.31 | 146.04 | -38.73 | 828.20 | attitude_penalty=-0.020 crash_penalty=-0.007 landing_bonus=1.095 progress_reward=0.037 | no_meaningful_improvement |
| 10 | attitude_penalty + contact_quality + fuel_penalty + progress_reward + time_penalty | 261.35 | 261.35 | 0.00 | 321.95 | attitude_penalty=-0.016 contact_quality=0.129 fuel_penalty=-0.018 progress_reward=0.090 time_penalty=-0.010 | target_solved_new_best |
