# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -105.90 | -105.90 | 0.00 | 72.00 | progress=0.016 soft_landing_proxy=0.002 stability_penalty=-0.014 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | 187.93 | 187.93 | 0.00 | 694.50 | progress=0.003 soft_landing_proxy=0.257 stability_penalty=-0.001 | new_best |
| 3 | progress + soft_landing_continuous + stability_penalty | 143.84 | 187.93 | -44.08 | 1000.00 | progress=0.003 soft_landing_continuous=0.237 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress + soft_landing_continuous + stability_penalty | 137.07 | 187.93 | -50.85 | 921.60 | progress=0.003 soft_landing_continuous=0.042 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress + soft_landing_proxy + stability_penalty | 144.59 | 187.93 | -43.34 | 1000.00 | progress=0.003 soft_landing_proxy=0.244 stability_penalty=-0.001 | unsolved_stagnation_fresh_restart |
| 6 | dist_reward + landing_proxy + stability_penalty | -113.31 | -113.31 | 0.00 | 71.90 | dist_reward=-0.972 landing_proxy=0.002 stability_penalty=-0.145 | new_best |
| 7 | landing_quality + progress + stability_penalty | 234.80 | 234.80 | 0.00 | 503.60 | landing_quality=0.485 progress=0.002 stability_penalty=-0.003 | target_solved_new_best |
| 8 | landing_quality + progress + stability_penalty + w_landing + w_progress | 181.99 | 234.80 | -52.81 | 916.10 | landing_quality=0.573 progress=0.002 stability_penalty=-0.003 w_landing=0.300 w_progress=0.900 | stop_after_solved_drop_keep_best |
