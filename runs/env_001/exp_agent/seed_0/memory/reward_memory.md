# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.10 | -107.10 | 0.00 | 74.20 | progress_delta_reward=0.016 soft_landing_proxy=0.005 stability_penalty=-0.142 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | -111.45 | -107.10 | -4.35 | 74.10 | progress_delta_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.058 | no_meaningful_improvement |
| 3 | angular_vel_penalty + energy_penalty + potential_shaping + soft_landing_proxy | -1771.11 | -107.10 | -1664.01 | 213.50 | angular_vel_penalty=-0.001 energy_penalty=-0.020 potential_shaping=0.130 soft_landing_proxy=0.001 | no_meaningful_improvement |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | 261.78 | 261.78 | 0.00 | 284.00 | progress_delta_reward=0.004 soft_landing_proxy=0.485 stability_penalty=-0.001 | target_solved_new_best |
| 5 | progress_delta_reward + soft_landing_proxy + stability_penalty | 191.83 | 261.78 | -69.95 | 825.70 | progress_delta_reward=0.003 soft_landing_proxy=0.661 stability_penalty=-0.000 | stop_after_solved_drop_keep_best |
