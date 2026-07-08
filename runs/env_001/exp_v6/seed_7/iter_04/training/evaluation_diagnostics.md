# Offline Policy Diagnostics

> Shadow-mode replay only. These statistics describe deterministic evaluation trajectories, not historical training trajectories.

## Replay

- env_id: LunarLander-v2
- episodes: 10
- seed_base: 10007
- reward_errors: 0

## Evaluation Summary

| metric | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|
| original_return | -109.076793 | 8.957799 | -108.213497 | -123.234285 | -95.059093 |
| generated_return | -70.931435 | 15.130269 | -76.015431 | -96.150286 | -47.573612 |
| episode_length | 68.300000 | 9.413288 | 68.000000 | 52.000000 | 82.000000 |

- termination_rate: 1.000000
- truncation_rate: 0.000000

## Additive-Term Inference

- status: exact
- reward_terms: angle_penalty, angular_vel_penalty, distance_reward, energy_penalty, time_penalty
- diagnostics/modulators: none
- mean_abs_residual: 0.000000

## Step Component Statistics

| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| angle_penalty | -0.021144 | 0.021144 | 1.000000 | -0.021144 | 0.021144 | -0.290898 | -0.000076 | 683 |
| angular_vel_penalty | -0.037277 | 0.037277 | 1.000000 | -0.037277 | 0.037277 | -5.834163 | -0.000000 | 683 |
| distance_reward | -0.967910 | 0.967910 | 1.000000 | -0.967910 | 0.967910 | -1.457063 | -0.048692 | 683 |
| energy_penalty | -0.002196 | 0.002196 | 0.043924 | -0.050000 | 0.050000 | -0.050000 | 0.000000 | 683 |
| generated_reward | -1.038528 | 1.038528 | 1.000000 | -1.038528 | 1.038528 | -6.614272 | -0.074956 | 683 |
| time_penalty | -0.010000 | 0.010000 | 1.000000 | -0.010000 | 0.010000 | -0.010000 | -0.010000 | 683 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| angle_penalty | -1.444139 | 0.624278 | -1.376154 | -2.468295 | -0.464294 | 10 |
| angular_vel_penalty | -2.546027 | 2.318602 | -1.238072 | -6.499189 | -0.615681 | 10 |
| distance_reward | -66.108269 | 13.831231 | -68.723506 | -87.417927 | -45.466940 | 10 |
| energy_penalty | -0.150000 | 0.089443 | -0.200000 | -0.300000 | 0.000000 | 10 |
| time_penalty | -0.683000 | 0.094133 | -0.680000 | -0.820000 | -0.520000 | 10 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10007 | -108.460728 | -96.150286 | 81 | True | False | terminated_unclassified | 0.386092 | 0.961564 | 0.012346 |
| 1 | 10008 | -119.790911 | -77.301055 | 76 | True | False | terminated_unclassified | 0.176610 | 0.043547 | 0.013158 |
| 2 | 10009 | -107.966266 | -80.049319 | 74 | True | False | terminated_unclassified | 0.476449 | 0.164416 | 0.027027 |
| 3 | 10010 | -95.059093 | -47.573612 | 52 | True | False | terminated_unclassified | 0.351395 | 0.000000 | 0.019231 |
| 4 | 10011 | -98.057243 | -57.605414 | 60 | True | False | terminated_unclassified | 0.373487 | 0.059115 | 0.033333 |
| 5 | 10012 | -123.234285 | -74.729807 | 67 | True | False | terminated_unclassified | 0.425678 | 0.837887 | 0.000000 |
| 6 | 10013 | -119.122994 | -79.405018 | 69 | True | False | terminated_unclassified | 0.535553 | 1.627691 | 0.000000 |
| 7 | 10014 | -103.791025 | -54.229656 | 60 | True | False | terminated_unclassified | 0.210273 | 0.316329 | 0.033333 |
| 8 | 10015 | -103.478759 | -85.986350 | 82 | True | False | terminated_unclassified | 0.094317 | 0.027880 | 0.012195 |
| 9 | 10016 | -111.806626 | -56.283828 | 62 | True | False | terminated_unclassified | 0.064706 | 0.000037 | 0.032258 |
