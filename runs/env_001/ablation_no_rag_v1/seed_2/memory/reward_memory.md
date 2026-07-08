# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stable_landing_reward | -37.29 | -37.29 | 0.00 | 1000.00 | progress_reward=0.003 stable_landing_reward=4.135 | new_best |
| 2 | progress_reward + stable_landing_reward | -43.99 | -37.29 | -6.70 | 101.20 | progress_reward=0.014 stable_landing_reward=0.138 | no_meaningful_improvement |
| 3 | progress_reward + stable_landing_reward | -82.73 | -37.29 | -45.44 | 858.20 | progress_reward=0.003 stable_landing_reward=4.262 | no_meaningful_improvement |
| 4 | progress_reward + stable_landing_reward | -47.11 | -37.29 | -9.82 | 1000.00 | progress_reward=0.002 stable_landing_reward=5.711 | same_skeleton_persistent_negative_fresh_restart |
| 5 | angle_penalty + potential_diff | -24.46 | -24.46 | 0.00 | 396.35 | angle_penalty=-0.013 potential_diff=0.012 | new_best |
| 6 | angle_penalty + potential_diff + success_bonus | -112.79 | -24.46 | -88.34 | 681.70 | angle_penalty=-0.018 potential_diff=0.008 success_bonus=41.547 | no_meaningful_improvement |
| 7 | angle_penalty + landing_quality_reward + potential_diff | 0.62 | 0.62 | 0.00 | 117.70 | angle_penalty=-0.015 landing_quality_reward=0.006 potential_diff=0.013 | new_best |
| 8 | angle_penalty + landing_reward + potential_diff | -150.57 | 0.62 | -151.19 | 1000.00 | angle_penalty=-0.008 landing_reward=0.846 potential_diff=0.001 | no_meaningful_improvement |
| 9 | angle_penalty + landing_quality_reward + potential_diff | -16.61 | 0.62 | -17.23 | 143.25 | angle_penalty=-0.017 landing_quality_reward=0.028 potential_diff=0.014 | same_skeleton_oscillation_fresh_restart |
| 10 | angle_penalty + approach_reward + landing_bounty | -109.75 | 0.62 | -110.37 | 68.45 | angle_penalty=-0.082 approach_reward=-2.805 landing_bounty=0.068 | no_meaningful_improvement |
