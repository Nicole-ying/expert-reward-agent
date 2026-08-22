# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_velocity_penalty | -0.111680 | 0.111680 | 1.000000 | -0.111680 | 0.111680 | -1.052524 | -0.000082 | 1003520 |
| component.forward_velocity_reward | 0.388398 | 0.655569 | 1.000000 | 0.388398 | 0.655569 | -4.762753 | 5.394861 | 1003520 |
| component.height_boundary_penalty | -0.033423 | 0.033423 | 0.434950 | -0.076843 | 0.076843 | -0.714264 | -0.000000 | 1003520 |
| component.lateral_velocity_penalty | -0.099726 | 0.099726 | 1.000000 | -0.099726 | 0.099726 | -0.965440 | -0.000000 | 1003520 |
| component.orientation_reward | 0.025545 | 0.282571 | 1.000000 | 0.025545 | 0.282571 | -0.300000 | 0.300000 | 1003520 |
| component.total_reward | 0.169113 | 0.771656 | 1.000000 | 0.169113 | 0.771656 | -5.695966 | 5.423815 | 1003520 |
| generated_reward | 0.169113 | 0.771656 | 1.000000 | 0.169113 | 0.771656 | -5.695966 | 5.423815 | 1003520 |
| original_env_reward | -0.693827 | 1.001694 | 1.000000 | -0.693827 | 1.001694 | -6.408070 | 5.314021 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_velocity_penalty | -17.267463 | 17.267463 | -121.198776 | -0.535162 | 6485 |
| forward_velocity_reward | 60.018158 | 63.215708 | -115.698759 | 1220.684330 | 6485 |
| height_boundary_penalty | -5.165151 | 5.165151 | -70.920803 | -0.200002 | 6485 |
| lateral_velocity_penalty | -15.419281 | 15.419281 | -127.955955 | -0.190430 | 6485 |
| orientation_reward | 3.964604 | 39.394399 | -276.984866 | 289.197124 | 6485 |
| total_reward | 26.130867 | 87.513929 | -571.581802 | 1270.928370 | 6485 |
