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
| original_return | 238.083812 | 42.162996 | 249.261594 | 109.652525 | 282.466855 |
| generated_return | 120.833201 | 196.617913 | 40.762975 | 28.715289 | 660.136042 |
| episode_length | 452.100000 | 223.948633 | 360.500000 | 336.000000 | 1000.000000 |

- termination_rate: 0.900000
- truncation_rate: 0.100000

## Additive-Term Inference

- status: exact
- reward_terms: landing_proxy, progress_reward, stability_penalty
- diagnostics/modulators: dist_gate
- mean_abs_residual: 0.000000

## Step Component Statistics

| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dist_gate | 0.490629 | 0.490629 | 1.000000 | 0.490629 | 0.490629 | 0.119665 | 0.994861 | 9042 |
| generated_reward | 0.267271 | 0.267395 | 1.000000 | 0.267271 | 0.267395 | -0.008041 | 0.997621 | 9042 |
| landing_proxy | 0.264982 | 0.264982 | 0.307454 | 0.861858 | 0.861858 | 0.000000 | 0.997626 | 9042 |
| progress_reward | 0.002981 | 0.003050 | 0.971688 | 0.003068 | 0.003139 | -0.007616 | 0.013980 | 9042 |
| stability_penalty | -0.000692 | 0.000692 | 1.000000 | -0.000692 | 0.000692 | -0.003121 | -0.000000 | 9042 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| dist_gate | 221.813241 | 171.561251 | 153.521129 | 138.459834 | 747.851528 | 20 |
| landing_proxy | 119.798312 | 196.638697 | 39.676926 | 27.734053 | 659.133979 | 20 |
| progress_reward | 1.347884 | 0.024926 | 1.348070 | 1.292182 | 1.398320 | 20 |
| stability_penalty | -0.312995 | 0.057855 | -0.301263 | -0.440128 | -0.239501 | 20 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10005 | 225.737849 | 41.321337 | 368 | True | False | terminated_unclassified | 0.050504 | 0.000000 | 0.114130 |
| 1 | 10006 | 254.092013 | 28.715289 | 352 | True | False | terminated_unclassified | 0.082326 | 0.000000 | 0.085227 |
| 2 | 10007 | 253.384139 | 47.014128 | 394 | True | False | terminated_unclassified | 0.061775 | 0.000000 | 0.124365 |
| 3 | 10008 | 221.490256 | 45.816270 | 378 | True | False | terminated_unclassified | 0.086844 | 0.000000 | 0.121693 |
| 4 | 10009 | 269.288320 | 32.233517 | 348 | True | False | terminated_unclassified | 0.054477 | 0.000000 | 0.094828 |
| 5 | 10010 | 282.466855 | 660.136042 | 951 | True | False | terminated_unclassified | 0.001419 | 0.000000 | 0.707676 |
| 6 | 10011 | 269.776098 | 39.542307 | 336 | True | False | terminated_unclassified | 0.056558 | 0.000000 | 0.119048 |
| 7 | 10012 | 260.683663 | 36.957467 | 355 | True | False | terminated_unclassified | 0.072921 | 0.000000 | 0.107042 |
| 8 | 10013 | 277.490532 | 31.710133 | 336 | True | False | terminated_unclassified | 0.086701 | 0.000000 | 0.086310 |
| 9 | 10014 | 109.652525 | 485.565986 | 1000 | False | True | timeout | 0.112284 | 0.001671 | 0.416000 |
| 10 | 10015 | 229.319321 | 31.077391 | 365 | True | False | terminated_unclassified | 0.055487 | 0.000000 | 0.084932 |
| 11 | 10016 | 223.135148 | 43.472329 | 356 | True | False | terminated_unclassified | 0.034837 | 0.000000 | 0.123596 |
| 12 | 10017 | 257.737990 | 35.597175 | 349 | True | False | terminated_unclassified | 0.070375 | 0.000000 | 0.106017 |
| 13 | 10018 | 245.139049 | 40.204613 | 349 | True | False | terminated_unclassified | 0.064275 | 0.000000 | 0.120344 |
| 14 | 10019 | 227.758744 | 39.384828 | 350 | True | False | terminated_unclassified | 0.043282 | 0.000000 | 0.114286 |
| 15 | 10020 | 149.158499 | 605.853825 | 1000 | False | True | timeout | 0.094621 | 0.000583 | 0.674000 |
| 16 | 10021 | 281.454935 | 44.994452 | 339 | True | False | terminated_unclassified | 0.028170 | 0.000000 | 0.135693 |
| 17 | 10022 | 236.116148 | 36.234895 | 373 | True | False | terminated_unclassified | 0.037111 | 0.000000 | 0.099196 |
| 18 | 10023 | 217.187122 | 44.001233 | 376 | True | False | terminated_unclassified | 0.066437 | 0.000000 | 0.122340 |
| 19 | 10024 | 270.607040 | 46.830801 | 367 | True | False | terminated_unclassified | 0.081916 | 0.000000 | 0.136240 |
