# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_bonus + stability_penalty | -108.32 | -108.32 | 0.00 | 68.50 | progress_reward=0.033 soft_landing_bonus=0.005 stability_penalty=-0.136 | new_best |
| 2 | progress_reward + soft_landing_bonus + stability_penalty | -106.27 | -106.27 | 0.00 | 68.55 | progress_reward=0.033 soft_landing_bonus=0.005 stability_penalty=-0.014 | new_best |
| 3 | progress_reward + soft_landing_bonus + stability_penalty | -107.99 | -106.27 | -1.72 | 68.50 | progress_reward=0.033 soft_landing_bonus=0.009 stability_penalty=-0.014 | no_meaningful_improvement |
| 4 | progress_reward + soft_landing_bonus + stability_penalty | -92.00 | -92.00 | 0.00 | 68.95 | progress_reward=0.033 soft_landing_bonus=0.012 stability_penalty=-0.014 | same_skeleton_persistent_negative_fresh_restart |
| 5 | progress_reward + soft_landing_proxy + speed_tracking_reward | 162.94 | 162.94 | 0.00 | 463.70 | progress_reward=0.004 soft_landing_proxy=2.229 speed_tracking_reward=-0.374 | new_best |
| 6 | landing_improvement + progress_reward + speed_tracking_reward | -111.07 | 162.94 | -274.01 | 69.20 | landing_improvement=0.278 progress_reward=0.016 speed_tracking_reward=-0.620 | no_meaningful_improvement |
| 7 | fuel_penalty + progress_reward + soft_landing_proxy + speed_tracking_reward | 144.11 | 162.94 | -18.83 | 305.30 | fuel_penalty=-0.006 progress_reward=0.004 soft_landing_proxy=2.876 speed_tracking_reward=-0.317 | no_meaningful_improvement |
| 8 | fuel_penalty + progress_reward + soft_landing_proxy + speed_tracking_reward | 144.11 | 162.94 | -18.83 | 305.30 | fuel_penalty=-0.006 progress_reward=0.004 soft_landing_proxy=2.876 speed_tracking_reward=-0.317 | unsolved_high_achievement_continue_from_best |
| 9 | landing_improvement + progress_reward + speed_tracking_reward | -108.87 | 162.94 | -271.81 | 68.40 | landing_improvement=0.152 progress_reward=0.016 speed_tracking_reward=-0.619 | no_meaningful_improvement |
| 10 | progress_reward + soft_landing_proxy + speed_tracking_reward | 180.21 | 180.21 | 0.00 | 553.30 | progress_reward=0.003 soft_landing_proxy=3.331 speed_tracking_reward=-0.292 | new_best |
