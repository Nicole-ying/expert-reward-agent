# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_bonus + stability_penalty | -110.97 | -110.97 | 0.00 | 68.40 | distance_reward=-0.974 soft_landing_bonus=0.002 stability_penalty=-0.127 | new_best |
| 2 | distance_reward + landing_quality + stability_penalty | -112.96 | -110.97 | -1.99 | 68.40 | distance_reward=-0.972 landing_quality=0.010 stability_penalty=-0.127 | no_meaningful_improvement |
| 3 | distance_reward + landing_quality + stability_penalty | -108.37 | -108.37 | 0.00 | 68.45 | distance_reward=-0.973 landing_quality=0.139 stability_penalty=-0.128 | new_best |
| 4 | distance_reward + landing_quality + stability_penalty | -109.79 | -108.37 | -1.42 | 68.35 | distance_reward=-0.972 landing_quality=0.140 stability_penalty=-0.052 | same_skeleton_persistent_negative_fresh_restart |
| 5 | progress_delta + weighted_stability_penalty | -109.22 | -108.37 | -0.85 | 68.55 | progress_delta=0.161 weighted_stability_penalty=-0.385 | no_meaningful_improvement |
| 6 | distance_reward + landing_quality + stability_penalty | -33.72 | -33.72 | 0.00 | 254.80 | distance_reward=-0.922 landing_quality=0.495 stability_penalty=-0.118 | new_best |
| 7 | distance_reward + landing_quality + stability_penalty | -108.69 | -33.72 | -74.96 | 68.45 | distance_reward=-0.972 landing_quality=0.241 stability_penalty=-0.125 | no_meaningful_improvement |
| 8 | distance_reward + landing_quality + stability_penalty | -34.07 | -33.72 | -0.35 | 550.55 | distance_reward=-0.646 landing_quality=0.529 stability_penalty=-0.110 | no_meaningful_improvement |
| 9 | distance_reward + landing_quality + stability_penalty | -111.98 | -33.72 | -78.25 | 68.40 | distance_reward=-0.744 landing_quality=0.073 stability_penalty=-0.127 | same_skeleton_persistent_negative_fresh_restart |
| 10 | shaping_reward | 247.36 | 247.36 | 0.00 | 361.00 | shaping_reward=0.003 | target_solved_new_best |
