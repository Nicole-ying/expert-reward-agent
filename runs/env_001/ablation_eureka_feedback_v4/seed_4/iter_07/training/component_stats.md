# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.angle_stability | -0.014220 | 0.014220 | 0.999925 | -0.014221 | 0.014221 | -7.655549 | -0.000000 | 1003520 |
| component.approach_progress | 0.008885 | 0.009610 | 0.999459 | 0.008889 | 0.009616 | -0.070858 | 0.081252 | 1003520 |
| component.grounded_quality | 0.155041 | 0.155041 | 0.540498 | 0.286849 | 0.286849 | 0.000000 | 0.300000 | 1003520 |
| component.landing_quality | 0.068146 | 0.068146 | 1.000000 | 0.068146 | 0.068146 | 0.012783 | 0.099962 | 1003520 |
| component.total_reward | 0.111366 | 0.308770 | 1.000000 | 0.111366 | 0.308770 | -8.085008 | 0.399949 | 1003520 |
| component.velocity_penalty | -0.106487 | 0.106487 | 0.403283 | -0.264050 | 0.264050 | -1.341277 | -0.000000 | 1003520 |
| generated_reward | 0.111366 | 0.308770 | 1.000000 | 0.111366 | 0.308770 | -8.085008 | 0.399949 | 1003520 |
| original_env_reward | -0.171855 | 1.415706 | 1.000000 | -0.171855 | 1.415706 | -100.000000 | 123.631264 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| angle_stability | -3.663875 | 3.663875 | -126.065422 | -0.009625 | 3893 |
| approach_progress | 2.287867 | 2.288769 | -0.745254 | 2.840216 | 3893 |
| grounded_quality | 39.871313 | 39.871313 | 0.000000 | 246.964227 | 3893 |
| landing_quality | 17.525237 | 17.525237 | 1.182579 | 90.235913 | 3893 |
| total_reward | 28.582188 | 74.436566 | -170.231472 | 317.081681 | 3893 |
| velocity_penalty | -27.438355 | 27.438355 | -55.519500 | -7.865706 | 3893 |
