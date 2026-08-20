# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -100.50 | -100.50 | 0.00 | 68.80 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.014 | new_best |
| 2 | progress_reward + soft_landing_proxy + stability_penalty | -94.32 | -94.32 | 0.00 | 73.70 | progress_reward=0.017 soft_landing_proxy=0.002 stability_penalty=-0.002 | new_best |
| 3 | approach_quality_reward + progress_reward + stability_penalty | 110.02 | 110.02 | 0.00 | 613.35 | approach_quality_reward=0.762 progress_reward=0.002 stability_penalty=-0.000 | new_best |
| 4 | landing_shaping_reward + progress_reward + stability_penalty | 240.08 | 240.08 | 0.00 | 231.55 | landing_shaping_reward=0.190 progress_reward=0.006 stability_penalty=-0.001 | target_solved_new_best |
| 5 | landing_contact_reward + landing_shaping_reward + progress_reward + stability_penalty | -47.91 | 240.08 | -287.99 | 85.30 | landing_contact_reward=0.011 landing_shaping_reward=0.344 progress_reward=0.015 stability_penalty=-0.001 | stop_after_solved_drop_keep_best |
