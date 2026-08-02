# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | failure_penalty + landing_proxy + progress + stability_penalty | -166.96 | -166.96 | 0.00 | 997.95 | failure_penalty=-0.005 landing_proxy=0.764 progress=0.001 stability_penalty=-0.008 | new_best |
| 2 | failure_penalty + landing_proxy + progress + stability_penalty | -6.02 | -6.02 | 0.00 | 455.45 | failure_penalty=-0.003 landing_proxy=0.173 progress=0.026 stability_penalty=-0.016 | new_best |
| 3 | action_penalty + failure_penalty + progress + soft_landing + stability_penalty + success_reward | -61.64 | -6.02 | -55.62 | 72.75 | action_penalty=-0.021 failure_penalty=-0.007 progress=0.045 soft_landing=0.001 stability_penalty=-0.010 | no_meaningful_improvement |
| 4 | action_penalty + failure_penalty + progress + soft_landing + stability_penalty + success_reward | -124.55 | -6.02 | -118.53 | 68.40 | action_penalty=-0.011 failure_penalty=-0.219 progress=0.047 soft_landing=0.001 stability_penalty=-0.017 | no_meaningful_improvement |
| 5 | action_penalty + progress + safety_penalty + soft_landing + stability_penalty + success_reward | -122.21 | -6.02 | -116.19 | 68.30 | action_penalty=-0.009 progress=0.048 safety_penalty=0.000 soft_landing=0.001 stability_penalty=-0.014 | unsolved_stagnation_fresh_restart |
| 6 | angle_penalty + angvel_penalty + landing_bonus + progress | -90.25 | -6.02 | -84.22 | 1000.00 | angle_penalty=-0.002 angvel_penalty=-0.002 landing_bonus=6.382 progress=0.046 | no_meaningful_improvement |
| 7 | angle_penalty + angvel_penalty + landing_bonus + progress | 150.81 | 150.81 | 0.00 | 813.70 | angle_penalty=-0.003 angvel_penalty=-0.001 landing_bonus=1.415 progress=0.022 | new_best |
| 8 | angle_penalty + angvel_penalty + landing_bonus + progress | 157.86 | 157.86 | 0.00 | 933.10 | angle_penalty=-0.002 angvel_penalty=-0.001 landing_bonus=0.521 progress=0.033 | new_best |
| 9 | angle_penalty + angvel_penalty + landing_bonus + progress | 245.37 | 245.37 | 0.00 | 264.00 | angle_penalty=-0.002 angvel_penalty=-0.005 landing_bonus=0.361 progress=0.090 | target_solved_new_best |
| 10 | angle_penalty + angvel_penalty + fuel_penalty + landing_bonus + progress | -112.95 | 245.37 | -358.32 | 68.40 | angle_penalty=-0.001 angvel_penalty=-0.012 fuel_penalty=-0.005 landing_bonus=0.006 progress=0.163 | stop_after_solved_drop_keep_best |
