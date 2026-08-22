# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.49 | -159.49 | 0.00 | 303.35 | contact_reward=0.000 orientation_penalty=-0.008 progress_reward=-0.002 vel_damping=-0.004 | new_best |
| 2 | approach_reward + contact_bonus + fuel_cost + landing_quality + orientation_cost + proximity | 40.02 | 40.02 | 0.00 | 850.35 | approach_reward=0.036 contact_bonus=0.373 fuel_cost=-0.018 landing_quality=0.645 orientation_cost=-0.019 | new_best |
| 3 | approach_reward + contact_bonus + fuel_cost + landing_quality + orientation_cost + proximity_reward | -45.14 | 40.02 | -85.16 | 1000.00 | approach_reward=0.039 contact_bonus=0.478 fuel_cost=-0.022 landing_quality=0.838 orientation_cost=-0.015 | no_meaningful_improvement |
| 4 | approach_reward + contact_bonus + fuel_cost + landing_quality + orientation_cost + success_bonus | -119.41 | 40.02 | -159.43 | 68.45 | approach_reward=0.158 contact_bonus=0.046 fuel_cost=-0.013 landing_quality=0.038 orientation_cost=-0.040 | no_meaningful_improvement |
| 5 | approach_reward + contact_bonus + drift_penalty + fuel_cost + landing_quality + orientation_cost | -48.75 | 40.02 | -88.77 | 101.80 | approach_reward=0.070 contact_bonus=0.268 drift_penalty=-0.144 fuel_cost=-0.022 landing_quality=0.211 | unsolved_high_achievement_continue_from_best |
| 6 | approach_reward + contact_bonus + drift_penalty + fuel_cost + landing_quality + near_speed_penalty | -114.28 | 40.02 | -154.30 | 68.45 | approach_reward=0.127 contact_bonus=0.096 drift_penalty=-0.153 fuel_cost=-0.012 landing_quality=0.057 | no_meaningful_improvement |
| 7 | approach_reward + contact_bonus + fuel_cost + landing_quality + orientation_cost + proximity | -112.76 | 40.02 | -152.78 | 69.80 | approach_reward=0.078 contact_bonus=0.049 fuel_cost=-0.017 landing_quality=0.049 orientation_cost=-0.028 | no_meaningful_improvement |
| 8 | approach_reward + contact_bonus + crash_penalty + fuel_cost + landing_quality + main_decel_bonus | -10.31 | 40.02 | -50.33 | 1000.00 | approach_reward=0.015 contact_bonus=0.566 crash_penalty=-0.005 fuel_cost=-0.022 landing_quality=0.990 | unsolved_high_achievement_continue_from_best |
| 9 | approach_reward + contact_bonus + descent_reward + fuel_cost + landing_quality + orientation_cost | 35.63 | 40.02 | -4.39 | 574.25 | approach_reward=0.026 contact_bonus=1.854 descent_reward=0.011 fuel_cost=-0.020 landing_quality=1.938 | no_meaningful_improvement |
| 10 | aligned_descent + approach + contact_bonus + fuel_cost + landing_quality + orientation_cost | 176.14 | 176.14 | 0.00 | 424.60 | aligned_descent=0.039 approach=0.081 contact_bonus=0.312 fuel_cost=-0.018 landing_quality=0.527 | new_best |
