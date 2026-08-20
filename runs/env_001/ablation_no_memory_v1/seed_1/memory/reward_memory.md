# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_reward + stability_penalty | -111.41 | -111.41 | 0.00 | 68.40 | distance_reward=-0.971 soft_landing_reward=0.004 stability_penalty=-0.115 | new_best |
| 2 | progress_delta + soft_landing_reward + stability_penalty | -109.40 | -109.40 | 0.00 | 68.50 | progress_delta=0.048 soft_landing_reward=0.005 stability_penalty=-0.111 | new_best |
| 3 | progress_delta + soft_landing_reward + stability_penalty | -106.67 | -106.67 | 0.00 | 68.50 | progress_delta=0.048 soft_landing_reward=0.004 stability_penalty=-0.066 | new_best |
| 4 | deceleration_bonus + progress_delta + stability_penalty | -110.81 | -106.67 | -4.13 | 68.45 | deceleration_bonus=0.035 progress_delta=0.048 stability_penalty=-0.066 | unsolved_stagnation_fresh_restart |
| 5 | landing_bonus + progress_delta + stability_penalty | -115.44 | -106.67 | -8.76 | 68.40 | landing_bonus=0.041 progress_delta=0.016 stability_penalty=-0.016 | no_meaningful_improvement |
| 6 | progress_delta + soft_landing_reward + stability_penalty | 252.56 | 252.56 | 0.00 | 314.70 | progress_delta=0.011 soft_landing_reward=0.570 stability_penalty=-0.026 | target_solved_new_best |
| 7 | fuel_penalty + progress_delta + soft_landing_reward + stability_penalty | -109.44 | 252.56 | -361.99 | 68.45 | fuel_penalty=-0.003 progress_delta=0.049 soft_landing_reward=0.006 stability_penalty=-0.067 | stop_after_solved_drop_keep_best |
