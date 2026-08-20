# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | initial | -70.35 | -70.35 | 0.0 | 1000.0 | initial | new_best |
| 2 | ang_penalty + dist_penalty + fuel_penalty + landing_reward + vel_penalty | 36.30 | 36.30 | 0.00 | 368.55 | ang_penalty=-0.029 dist_penalty=-0.168 fuel_penalty=-0.014 landing_reward=0.713 vel_penalty=-0.091 | new_best |
| 3 | ang_penalty + fuel_penalty + landing_reward + progress_reward + vel_penalty | -109.41 | 36.30 | -145.71 | 68.45 | ang_penalty=-0.001 fuel_penalty=-0.002 landing_reward=0.011 progress_reward=0.008 vel_penalty=-0.013 | no_meaningful_improvement |
| 4 | ang_change + fuel_penalty + landing_change + progress_reward + vel_change | -114.95 | 36.30 | -151.25 | 68.40 | ang_change=-0.000 fuel_penalty=-0.003 landing_change=0.029 progress_reward=0.008 vel_change=-0.000 | no_meaningful_improvement |
| 5 | angle_penalty + contact_bonus + dist_reward + fuel_penalty + step_penalty + vel_penalty | -112.22 | 36.30 | -148.53 | 68.40 | angle_penalty=-0.007 contact_bonus=0.011 dist_reward=0.027 fuel_penalty=-0.004 step_penalty=-0.100 | unsolved_high_achievement_continue_from_best |
| 6 | angle_penalty + contact_bonus + dist_reward + fuel_penalty + step_penalty + vel_penalty | -117.69 | 36.30 | -154.00 | 68.30 | angle_penalty=-0.017 contact_bonus=0.063 dist_reward=0.127 fuel_penalty=-0.004 step_penalty=-0.020 | no_meaningful_improvement |
| 7 | angle_penalty + dist_reward + fuel_penalty + step_penalty + success_bonus + touch_bonus | -117.50 | 36.30 | -153.80 | 68.35 | angle_penalty=-0.006 dist_reward=0.160 fuel_penalty=-0.005 step_penalty=-0.020 success_bonus=0.479 | no_meaningful_improvement |
| 8 | contact_bonus + fuel_penalty + shaping_reward + success_bonus | -105.80 | 36.30 | -142.10 | 98.70 | contact_bonus=0.004 fuel_penalty=-0.007 shaping_reward=0.014 success_bonus=1.047 | unsolved_high_achievement_continue_from_best |
| 9 | angle_reward + contact_reward + fuel_penalty + proximity_reward + speed_reward + success_bonus | -63.45 | 36.30 | -99.75 | 1000.00 | angle_reward=0.642 contact_reward=0.124 fuel_penalty=-0.008 proximity_reward=0.623 speed_reward=0.761 | no_meaningful_improvement |
| 10 | contact_reward + fuel_penalty + progress_reward + proximity_reward + settle_reward | -200.97 | 36.30 | -237.27 | 1000.00 | contact_reward=1.297 fuel_penalty=-0.041 progress_reward=0.000 proximity_reward=8.851 settle_reward=0.751 | no_meaningful_improvement |
