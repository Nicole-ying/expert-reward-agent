# Offline Policy Diagnostics

> Shadow-mode replay only. These statistics describe deterministic evaluation trajectories, not historical training trajectories.

## Replay

- env_id: LunarLander-v2
- episodes: 20
- seed_base: 10005
- reward_errors: 0

## Evaluation Summary

| metric | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|
| original_return | -94.010644 | 21.610062 | -100.147328 | -122.559656 | -52.225025 |
| generated_return | 0.768316 | 0.578496 | 0.443380 | -0.006933 | 1.617540 |
| episode_length | 68.900000 | 9.710304 | 68.500000 | 54.000000 | 85.000000 |

- termination_rate: 1.000000
- truncation_rate: 0.000000

## Additive-Term Inference

- status: exact
- reward_terms: landing_proxy, progress_reward, stability_penalty
- diagnostics/modulators: none
- mean_abs_residual: 0.000000

## Step Component Statistics

| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| generated_reward | 0.011151 | 0.014215 | 1.000000 | 0.011151 | 0.014215 | -0.056993 | 0.918028 | 1378 |
| landing_proxy | 0.005926 | 0.005926 | 0.012337 | 0.480325 | 0.480325 | 0.000000 | 0.932957 | 1378 |
| progress_reward | 0.016297 | 0.016887 | 1.000000 | 0.016297 | 0.016887 | -0.014926 | 0.039039 | 1378 |
| stability_penalty | -0.011071 | 0.011071 | 1.000000 | -0.011071 | 0.011071 | -0.047151 | -0.000002 | 1378 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| landing_proxy | 0.408276 | 0.363543 | 0.338756 | 0.000000 | 0.932957 | 20 |
| progress_reward | 1.122839 | 0.158449 | 1.093461 | 0.885348 | 1.366795 | 20 |
| stability_penalty | -0.762800 | 0.079470 | -0.742015 | -0.892280 | -0.656533 | 20 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10005 | -115.431075 | 1.285828 | 74 | True | False | terminated_unclassified | 0.197055 | 0.000000 | 0.027027 |
| 1 | 10006 | -63.265378 | 0.441018 | 82 | True | False | terminated_unclassified | 0.424853 | 0.166898 | 0.036585 |
| 2 | 10007 | -110.048202 | 0.188128 | 81 | True | False | terminated_unclassified | 0.386380 | 0.968975 | 0.012346 |
| 3 | 10008 | -117.722997 | 1.358222 | 76 | True | False | terminated_unclassified | 0.174699 | 0.000000 | 0.026316 |
| 4 | 10009 | -65.852623 | 0.418427 | 76 | True | False | terminated_unclassified | 0.481767 | 0.242566 | 0.026316 |
| 5 | 10010 | -59.082007 | 0.365635 | 54 | True | False | terminated_unclassified | 0.357078 | 0.722631 | 0.018519 |
| 6 | 10011 | -52.225025 | 0.433732 | 62 | True | False | terminated_unclassified | 0.378779 | 0.204853 | 0.016129 |
| 7 | 10012 | -95.741347 | 0.273647 | 68 | True | False | terminated_unclassified | 0.428163 | 0.717935 | 0.000000 |
| 8 | 10013 | -116.981061 | -0.006933 | 68 | True | False | terminated_unclassified | 0.525513 | 1.634764 | 0.000000 |
| 9 | 10014 | -95.479637 | 1.014726 | 60 | True | False | terminated_unclassified | 0.209294 | 0.314465 | 0.016667 |
| 10 | 10015 | -104.553310 | 1.524182 | 82 | True | False | terminated_unclassified | 0.097166 | 0.000000 | 0.024390 |
| 11 | 10016 | -105.875614 | 1.572165 | 62 | True | False | terminated_unclassified | 0.061255 | 0.054509 | 0.032258 |
| 12 | 10017 | -91.864816 | 0.445742 | 54 | True | False | terminated_unclassified | 0.257479 | 0.974977 | 0.000000 |
| 13 | 10018 | -71.027532 | 1.428895 | 55 | True | False | terminated_unclassified | 0.113509 | 0.086679 | 0.036364 |
| 14 | 10019 | -105.323659 | 1.127902 | 69 | True | False | terminated_unclassified | 0.179661 | 0.212318 | 0.014493 |
| 15 | 10020 | -122.559656 | 0.065314 | 71 | True | False | terminated_unclassified | 0.466828 | 1.528645 | 0.014085 |
| 16 | 10021 | -73.249116 | 0.224344 | 60 | True | False | terminated_unclassified | 0.436246 | 0.894612 | 0.016667 |
| 17 | 10022 | -90.142321 | 1.476113 | 85 | True | False | terminated_unclassified | 0.052319 | 0.131134 | 0.023529 |
| 18 | 10023 | -109.527770 | 1.617540 | 77 | True | False | terminated_unclassified | 0.066473 | 0.000022 | 0.025974 |
| 19 | 10024 | -114.259725 | 0.111688 | 62 | True | False | terminated_unclassified | 0.448690 | 1.349450 | 0.016129 |
