# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_bonus + stability_penalty | -108.94 | -108.94 | 0.00 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.147 | new_best |
| 2 | progress_delta + soft_landing_bonus + stability_penalty | -110.57 | -108.94 | -1.62 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.006 | no_meaningful_improvement |
| 3 | bounded_proximity + soft_landing_bonus + stability_penalty | 251.94 | 251.94 | 0.00 | 363.30 | bounded_proximity=0.560 soft_landing_bonus=0.178 stability_penalty=0.003 | target_solved_new_best |
| 4 | bounded_proximity + soft_landing_bonus + stability_penalty | 235.96 | 251.94 | -15.97 | 478.90 | bounded_proximity=0.559 soft_landing_bonus=0.156 stability_penalty=0.003 | target_solved_no_improvement |
| 5 | bounded_proximity + soft_landing_bonus + stability_penalty | 255.82 | 251.94 | 3.89 | 440.40 | bounded_proximity=0.569 soft_landing_bonus=0.084 stability_penalty=0.003 | stop_solved_no_improvement_keep_best |
