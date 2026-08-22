# Reward Component Training Statistics

- steps_seen: 1001472
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angular_velocity_penalty | -0.113509 | 0.113509 | 0.999994 | -0.113510 | 0.113510 | -2.115900 | -0.000000 | 1001472 |
| component.gated_forward | 0.599554 | 0.779725 | 1.000000 | 0.599554 | 0.779725 | -8.002577 | 10.207128 | 1001472 |
| component.height_hinge_penalty | -0.184799 | 0.184799 | 0.467572 | -0.395231 | 0.395231 | -3.613179 | -0.000000 | 1001472 |
| component.lateral_penalty | -0.142836 | 0.142836 | 0.999773 | -0.142869 | 0.142869 | -10.808322 | -0.000000 | 1001472 |
| component.total_reward | 0.158409 | 0.948054 | 1.000000 | 0.158409 | 0.948054 | -12.085073 | 9.334401 | 1001472 |
| generated_reward | 0.158409 | 0.948054 | 1.000000 | 0.158409 | 0.948054 | -12.085073 | 9.334401 | 1001472 |
| original_env_reward | -1.182410 | 1.338208 | 1.000000 | -1.182410 | 1.338208 | -5.961632 | 4.347311 | 1001472 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angular_velocity_penalty | -31.336686 | 31.336686 | -253.739678 | -0.576889 | 3627 |
| gated_forward | 165.489070 | 173.010065 | -214.360204 | 2441.287271 | 3627 |
| height_hinge_penalty | -51.025930 | 51.025930 | -395.035642 | 0.000000 | 3627 |
| lateral_penalty | -39.431310 | 39.431310 | -287.829238 | -0.148300 | 3627 |
| total_reward | 43.695144 | 173.689908 | -753.803826 | 2145.517571 | 3627 |
