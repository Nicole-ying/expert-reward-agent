# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | approach_reward + soft_landing_proxy + stability_penalty | -95.25 | -95.25 | 0.00 | 68.80 | approach_reward=0.016 soft_landing_proxy=0.025 stability_penalty=-0.019 | new_best |
| 2 | approach_reward + soft_landing_proxy + stability_penalty | 149.44 | 149.44 | 0.00 | 564.95 | approach_reward=0.005 soft_landing_proxy=1.728 stability_penalty=-0.002 | new_best |
| 3 | approach_reward + soft_landing_proxy + stability_penalty | -10.51 | 149.44 | -159.95 | 1000.00 | approach_reward=0.088 soft_landing_proxy=1.673 stability_penalty=-0.002 | no_meaningful_improvement |
| 4 | approach_reward + soft_landing_proxy + stability_penalty | 136.73 | 149.44 | -12.71 | 237.45 | approach_reward=0.027 soft_landing_proxy=0.690 stability_penalty=-0.002 | same_skeleton_oscillation_fresh_restart |
| 5 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 6 | landing_reward + shaping_reward | -236.15 | 149.44 | -385.59 | 835.50 | landing_reward=0.002 shaping_reward=0.951 | no_meaningful_improvement |
| 7 | approach_reward + soft_landing_proxy + stability_penalty | 198.14 | 198.14 | 0.00 | 406.05 | approach_reward=0.003 soft_landing_proxy=1.691 stability_penalty=-0.002 | new_best |
| 8 | approach_reward + engine_penalty + soft_landing_proxy + stability_penalty | -7.39 | 198.14 | -205.53 | 1000.00 | approach_reward=0.004 engine_penalty=-0.004 soft_landing_proxy=1.661 stability_penalty=-0.002 | no_meaningful_improvement |
| 9 | approach_reward + contact_bonus + proximity_quality + stability_penalty | -33.97 | 198.14 | -232.12 | 1000.00 | approach_reward=0.002 contact_bonus=1.019 proximity_quality=0.845 stability_penalty=-0.002 | no_meaningful_improvement |
| 10 | approach_reward + soft_landing_proxy + stability_penalty | 221.75 | 221.75 | 0.00 | 520.35 | approach_reward=0.003 soft_landing_proxy=1.122 stability_penalty=-0.001 | target_solved_new_best |
