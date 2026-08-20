# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + stability_penalty | -112.57 | -112.57 | 0.00 | 70.60 | progress_delta=0.016 stability_penalty=-0.014 | new_best |
| 2 | progress_delta + stability_penalty | -96.19 | -96.19 | 0.00 | 99.80 | progress_delta=0.015 stability_penalty=-0.001 | new_best |
| 3 | progress_delta + stability_penalty | -101.92 | -96.19 | -5.73 | 86.80 | progress_delta=0.014 stability_penalty=-0.007 | no_meaningful_improvement |
| 4 | angle_reduction + angvel_reduction + progress_delta + speed_reduction | 84.80 | 84.80 | 0.00 | 823.30 | angle_reduction=-0.000 angvel_reduction=-0.000 progress_delta=0.004 speed_reduction=0.000 | new_best |
| 5 | angle_reduction + angvel_reduction + landing_incentive + progress_delta + speed_reduction | 147.94 | 147.94 | 0.00 | 1000.00 | angle_reduction=-0.000 angvel_reduction=-0.000 landing_incentive=0.298 progress_delta=0.002 speed_reduction=0.000 | new_best |
| 6 | angle_reduction + angvel_reduction + landing_incentive + progress_delta + speed_reduction | 260.64 | 260.64 | 0.00 | 314.40 | angle_reduction=-0.000 angvel_reduction=-0.000 landing_incentive=0.508 progress_delta=0.002 speed_reduction=0.000 | target_solved_new_best |
| 7 | landing_incentive + potential_reward | 256.27 | 260.64 | -4.37 | 349.60 | landing_incentive=0.428 potential_reward=0.002 | target_solved_no_improvement |
| 8 | angle_reduction + angvel_reduction + landing_incentive + progress_delta + speed_reduction | 142.95 | 260.64 | -117.69 | 1000.00 | angle_reduction=-0.000 angvel_reduction=-0.000 landing_incentive=0.392 progress_delta=0.002 speed_reduction=0.000 | stop_after_solved_drop_keep_best |
