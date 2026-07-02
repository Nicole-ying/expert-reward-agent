# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | -111.12 | -111.44 | 0.33 | 71.40 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.006 | no_meaningful_improvement |
| 4 | potential_shaping_reward + stability_penalty | -111.89 | -111.89 | 0.00 | 71.30 | potential_shaping_reward=0.026 stability_penalty=-0.068 | new_best |
| 5 | shaping_reward + stability_penalty | -81.52 | -81.52 | 0.00 | 98.10 | shaping_reward=0.014 stability_penalty=-0.006 | new_best |
| 6 | shaping_reward | 263.61 | 263.61 | 0.00 | 350.10 | shaping_reward=0.047 | target_solved_new_best |
| 7 | shaping_reward | 263.61 | 263.61 | 0.00 | 350.10 | shaping_reward=0.047 | target_solved_no_improvement |
| 8 | shaping_reward | 263.61 | 263.61 | 0.00 | 350.10 | shaping_reward=0.047 | stop_solved_no_improvement_keep_best |
