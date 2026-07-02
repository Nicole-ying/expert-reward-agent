# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
| 3 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -12.26 | -12.26 | 0.00 | 1000.00 | approach_bonus=2.711 distance_penalty=-0.003 progress_delta_reward=0.002 stability_penalty=-0.005 | new_best |
| 4 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -60.67 | -12.26 | -48.41 | 85.30 | approach_bonus=0.005 distance_penalty=-0.004 progress_delta_reward=0.013 stability_penalty=-0.007 | no_meaningful_improvement |
| 5 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -110.31 | -12.26 | -98.05 | 72.00 | approach_bonus=0.009 distance_penalty=-0.005 progress_delta_reward=0.016 stability_penalty=-0.008 | no_meaningful_improvement |
| 6 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -55.92 | -12.26 | -43.65 | 955.10 | approach_bonus=3.129 distance_penalty=-0.002 progress_delta_reward=0.002 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 7 | potential_shaping + soft_landing_proxy + stability_penalty | -108.75 | -108.75 | 0.00 | 72.00 | potential_shaping=0.021 soft_landing_proxy=0.010 stability_penalty=-0.031 | new_best |
| 8 | potential_shaping + soft_landing_proxy + stability_penalty | -0.83 | -0.83 | 0.00 | 1000.00 | potential_shaping=0.005 soft_landing_proxy=0.917 stability_penalty=-0.015 | new_best |
| 9 | landing_quality + progress_signal + proximity + quality_signal + stability_penalty | 251.36 | 251.36 | 0.00 | 375.70 | landing_quality=0.319 progress_signal=1.187 proximity=0.461 quality_signal=0.204 stability_penalty=-0.002 | target_solved_new_best |
| 10 | landing_quality + progress_signal + proximity + quality_signal + stability_penalty | 35.56 | 251.36 | -215.80 | 1000.00 | landing_quality=0.576 progress_signal=1.241 proximity=0.482 quality_signal=0.244 stability_penalty=-0.003 | stop_after_solved_drop_keep_best |
