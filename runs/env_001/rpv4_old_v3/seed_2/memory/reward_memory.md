# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | -155.98 | -155.98 | 0.00 | 154.40 | action_cost=-0.013 angle_penalty=-0.015 landing_velocity_penalty=-0.118 progress_shaping=0.035 | new_best |
| 2 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -119.51 | -119.51 | 0.00 | 68.30 | action_cost=-0.006 angle_penalty=-0.004 progress_shaping=0.080 success_landing_bonus=0.208 | new_best |
| 3 | action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | -203.45 | -119.51 | -83.94 | 154.15 | action_cost=-0.015 angle_penalty=-0.034 crash_penalty=-0.425 progress_shaping=-0.007 success_landing_bonus=0.002 | no_meaningful_improvement |
| 4 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -114.38 | -114.38 | 0.00 | 68.45 | action_cost=-0.008 angle_penalty=-0.004 progress_shaping=0.079 success_landing_bonus=0.449 | new_best |
| 5 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -46.51 | -46.51 | 0.00 | 740.15 | action_cost=-0.014 angle_penalty=-0.022 progress_shaping=0.040 success_landing_bonus=28.088 | new_best |
| 6 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -111.18 | -46.51 | -64.67 | 68.45 | action_cost=-0.002 angle_penalty=-0.002 progress_shaping=0.081 success_landing_bonus=0.005 | no_meaningful_improvement |
| 7 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -98.93 | -46.51 | -52.42 | 69.60 | action_cost=-0.003 angle_penalty=-0.004 progress_shaping=0.081 success_landing_bonus=0.023 | same_skeleton_persistent_negative_fresh_restart |
| 8 | angle_stab + fuel_penalty + progress + vel_damping | 26.45 | 26.45 | 0.00 | 206.50 | angle_stab=-0.003 fuel_penalty=-0.006 progress=0.014 vel_damping=-0.011 | new_best |
| 9 | angle_stab + fuel_penalty + progress + success_landing_bonus + vel_damping | 145.17 | 145.17 | 0.00 | 718.80 | angle_stab=-0.004 fuel_penalty=-0.008 progress=0.006 success_landing_bonus=5.779 vel_damping=-0.008 | new_best |
| 10 | angle_stab + fuel_penalty + progress + success_landing_bonus + vel_damping | 131.35 | 145.17 | -13.82 | 831.25 | angle_stab=-0.006 fuel_penalty=-0.008 progress=0.101 success_landing_bonus=4.458 vel_damping=-0.016 | no_meaningful_improvement |
| 11 | angle_stab + fuel_penalty + landing_quality + progress + vel_damping | -7.59 | 145.17 | -152.76 | 1000.00 | angle_stab=-0.005 fuel_penalty=-0.007 landing_quality=1.614 progress=0.062 vel_damping=-0.007 | no_meaningful_improvement |
