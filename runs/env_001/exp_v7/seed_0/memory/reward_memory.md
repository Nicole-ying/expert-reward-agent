# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
| 2 | distance_reward + soft_landing_continuous + stability_penalty | -115.29 | -111.87 | -3.42 | 68.40 | distance_reward=-0.969 soft_landing_continuous=0.013 stability_penalty=-0.027 | no_meaningful_improvement |
| 3 | contact_bonus + potential_shaping | -1368.99 | -111.87 | -1257.12 | 177.60 | contact_bonus=0.001 potential_shaping=0.126 | no_meaningful_improvement |
| 4 | distance_reward + soft_landing + stability_penalty | -113.14 | -111.87 | -1.27 | 68.40 | distance_reward=-0.971 soft_landing=0.012 stability_penalty=-0.025 | unsolved_stagnation_fresh_restart |
| 5 | progress_reward + soft_landing_reward + stability_penalty | 216.84 | 216.84 | 0.00 | 527.15 | progress_reward=0.002 soft_landing_reward=0.336 stability_penalty=-0.001 | target_solved_new_best |
