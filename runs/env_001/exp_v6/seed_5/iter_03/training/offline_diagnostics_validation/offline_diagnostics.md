# Offline Policy Diagnostics

> Shadow-mode replay only. These statistics describe deterministic evaluation trajectories, not historical training trajectories.

## Replay

- env_id: LunarLander-v2
- episodes: 10
- seed_base: 10005
- reward_errors: 0

## Evaluation Summary

| metric | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|
| original_return | 242.406225 | 48.205305 | 257.387838 | 109.652525 | 282.466855 |
| generated_return | 144.901248 | 217.578874 | 40.431822 | 28.715289 | 660.136042 |
| episode_length | 481.800000 | 247.680762 | 361.500000 | 336.000000 | 1000.000000 |

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
| dist_gate | 0.517412 | 0.517412 | 1.000000 | 0.517412 | 0.517412 | 0.120711 | 0.994861 | 4818 |
| generated_reward | 0.300750 | 0.300898 | 1.000000 | 0.300750 | 0.300898 | -0.007240 | 0.997621 | 4818 |
| landing_proxy | 0.298679 | 0.298679 | 0.355957 | 0.839089 | 0.839089 | 0.000000 | 0.997626 | 4818 |
| progress_reward | 0.002790 | 0.002856 | 0.972603 | 0.002868 | 0.002937 | -0.006477 | 0.013980 | 4818 |
| stability_penalty | -0.000719 | 0.000719 | 1.000000 | -0.000719 | 0.000719 | -0.002566 | -0.000002 | 4818 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| dist_gate | 249.289094 | 205.569762 | 150.614009 | 138.459834 | 747.851528 | 10 |
| landing_proxy | 143.903686 | 217.594413 | 39.369579 | 27.734053 | 659.133979 | 10 |
| progress_reward | 1.343993 | 0.026862 | 1.342760 | 1.292182 | 1.398320 | 10 |
| stability_penalty | -0.346432 | 0.054322 | -0.349404 | -0.440128 | -0.259588 | 10 |

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
