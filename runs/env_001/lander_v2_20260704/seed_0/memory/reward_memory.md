# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -518.69 | -518.69 | 0.00 | 62.35 | distance_reward=-0.981 soft_landing_proxy=0.000 stability_penalty=-0.035 | new_best |
| 2 | distance_reward + soft_landing_proxy + stability_penalty | -85.75 | -85.75 | 0.00 | 88.95 | distance_reward=0.071 soft_landing_proxy=0.005 stability_penalty=-0.014 | new_best |
| 3 | distance_reward + landing_quality + stability_penalty | -17.84 | -17.84 | 0.00 | 1000.00 | distance_reward=0.016 landing_quality=1.007 stability_penalty=-0.004 | new_best |
| 4 | distance_reward + landing_quality + stability_penalty | -110.72 | -17.84 | -92.88 | 68.50 | distance_reward=0.080 landing_quality=0.092 stability_penalty=-0.015 | no_meaningful_improvement |
| 5 | distance_reward + landing_quality + stability_penalty | -23.32 | -17.84 | -5.47 | 1000.00 | distance_reward=0.017 landing_quality=2.090 stability_penalty=-0.004 | no_meaningful_improvement |
| 6 | distance_reward + landing_quality + stability_penalty | -73.82 | -17.84 | -55.98 | 87.30 | distance_reward=0.073 landing_quality=0.025 stability_penalty=-0.014 | unsolved_stagnation_fresh_restart |
| 7 | angle_penalty + contact_reward + proximity_reward + velocity_penalty | 167.99 | 167.99 | 0.00 | 647.40 | angle_penalty=-0.006 contact_reward=0.267 proximity_reward=0.723 velocity_penalty=-0.024 | new_best |
| 8 | angle_penalty + contact_reward + proximity_improvement + velocity_penalty | 215.30 | 215.30 | 0.00 | 577.65 | angle_penalty=-0.009 contact_reward=0.306 proximity_improvement=0.156 velocity_penalty=-0.037 | target_solved_new_best |
| 9 | angle_penalty + contact_reward + proximity_improvement + velocity_penalty | 50.78 | 215.30 | -164.51 | 632.75 | angle_penalty=-0.005 contact_reward=2.747 proximity_improvement=0.226 velocity_penalty=-0.071 | stop_after_solved_drop_keep_best |
