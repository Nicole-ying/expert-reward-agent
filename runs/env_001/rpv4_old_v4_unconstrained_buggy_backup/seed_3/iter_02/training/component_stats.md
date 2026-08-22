# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.contact_bonus | 7.304986 | 7.304986 | 0.073050 | 100.000000 | 100.000000 | 0.000000 | 100.000000 | 1003520 |
| component.goal_proximity | -0.041808 | 0.041808 | 1.000000 | -0.041808 | 0.041808 | -0.084661 | -0.000007 | 1003520 |
| component.orientation_penalty | -0.005599 | 0.005599 | 1.000000 | -0.005599 | 0.005599 | -0.363015 | -0.000000 | 1003520 |
| component.total_reward | 1.342479 | 1.579515 | 1.000000 | 1.342479 | 1.579515 | -0.599425 | 20.000000 | 1003520 |
| component.velocity_penalty | -0.074071 | 0.074071 | 0.998399 | -0.074190 | 0.074190 | -0.486199 | -0.000000 | 1003520 |
| generated_reward | 1.342479 | 1.579515 | 1.000000 | 1.342479 | 1.579515 | -0.599425 | 20.000000 | 1003520 |
| original_env_reward | -0.730981 | 2.944359 | 1.000000 | -0.730981 | 2.944359 | -100.000000 | 128.588741 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| contact_bonus | 679.907903 | 679.907903 | 0.000000 | 81600.000000 | 10641 |
| goal_proximity | -3.940168 | 3.940168 | -39.034607 | -2.050034 | 10641 |
| orientation_penalty | -0.527437 | 0.527437 | -20.407963 | -0.016036 | 10641 |
| total_reward | 124.808078 | 132.635622 | -56.491147 | 16311.171581 | 10641 |
| velocity_penalty | -6.983892 | 6.983892 | -13.363558 | -3.418823 | 10641 |
