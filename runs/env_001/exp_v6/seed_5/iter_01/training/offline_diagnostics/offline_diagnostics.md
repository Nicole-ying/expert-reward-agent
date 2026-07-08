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
| original_return | -111.215774 | 8.638911 | -112.279245 | -128.042662 | -97.067314 |
| generated_return | 0.711236 | 0.664398 | 0.381992 | -0.048686 | 1.685120 |
| episode_length | 68.250000 | 9.771771 | 68.500000 | 52.000000 | 85.000000 |

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
| generated_reward | 0.010421 | 0.013716 | 1.000000 | 0.010421 | 0.013716 | -0.055921 | 0.998709 | 1365 |
| landing_proxy | 0.005128 | 0.005128 | 0.005128 | 1.000000 | 1.000000 | 0.000000 | 1.000000 | 1365 |
| progress_reward | 0.016489 | 0.017068 | 1.000000 | 0.016489 | 0.017068 | -0.015163 | 0.039081 | 1365 |
| stability_penalty | -0.011196 | 0.011196 | 1.000000 | -0.011196 | 0.011196 | -0.046032 | -0.000002 | 1365 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| landing_proxy | 0.350000 | 0.476970 | 0.000000 | 0.000000 | 1.000000 | 20 |
| progress_reward | 1.125390 | 0.157068 | 1.098106 | 0.877645 | 1.367557 | 20 |
| stability_penalty | -0.764154 | 0.082974 | -0.717828 | -0.926331 | -0.659720 | 20 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10005 | -113.170281 | 1.528614 | 74 | True | False | terminated_unclassified | 0.196273 | 0.000000 | 0.027027 |
| 1 | 10006 | -116.945633 | 0.120292 | 78 | True | False | terminated_unclassified | 0.411474 | 1.048729 | 0.025641 |
| 2 | 10007 | -113.990450 | 0.176559 | 81 | True | False | terminated_unclassified | 0.385234 | 0.997683 | 0.012346 |
| 3 | 10008 | -115.273081 | 1.542321 | 76 | True | False | terminated_unclassified | 0.174082 | 0.000000 | 0.026316 |
| 4 | 10009 | -113.915124 | 0.083714 | 74 | True | False | terminated_unclassified | 0.473668 | 0.163842 | 0.027027 |
| 5 | 10010 | -97.868960 | 0.330909 | 52 | True | False | terminated_unclassified | 0.349711 | 0.000000 | 0.019231 |
| 6 | 10011 | -100.410691 | 0.251836 | 60 | True | False | terminated_unclassified | 0.371661 | 0.127111 | 0.033333 |
| 7 | 10012 | -125.885489 | 0.191846 | 67 | True | False | terminated_unclassified | 0.424932 | 0.797768 | 0.000000 |
| 8 | 10013 | -114.861641 | -0.048686 | 69 | True | False | terminated_unclassified | 0.533216 | 1.606698 | 0.000000 |
| 9 | 10014 | -104.784936 | 1.500267 | 60 | True | False | terminated_unclassified | 0.208920 | 0.282847 | 0.033333 |
| 10 | 10015 | -103.314607 | 0.623732 | 82 | True | False | terminated_unclassified | 0.096671 | 0.000000 | 0.012195 |
| 11 | 10016 | -111.388210 | 1.685120 | 62 | True | False | terminated_unclassified | 0.061145 | 0.054673 | 0.032258 |
| 12 | 10017 | -103.810338 | 0.433074 | 54 | True | False | terminated_unclassified | 0.255554 | 1.016441 | 0.000000 |
| 13 | 10018 | -103.039674 | 0.594008 | 54 | True | False | terminated_unclassified | 0.111926 | 1.201152 | 0.018519 |
| 14 | 10019 | -128.042662 | 1.534995 | 68 | True | False | terminated_unclassified | 0.179309 | 0.244601 | 0.014706 |
| 15 | 10020 | -122.033815 | 0.069632 | 71 | True | False | terminated_unclassified | 0.465444 | 1.500288 | 0.014085 |
| 16 | 10021 | -108.759118 | 0.175852 | 59 | True | False | terminated_unclassified | 0.430590 | 0.925653 | 0.016949 |
| 17 | 10022 | -97.067314 | 1.651019 | 85 | True | False | terminated_unclassified | 0.051557 | 0.130033 | 0.023529 |
| 18 | 10023 | -110.270183 | 1.678549 | 77 | True | False | terminated_unclassified | 0.066996 | 0.000045 | 0.025974 |
| 19 | 10024 | -119.483278 | 0.101057 | 62 | True | False | terminated_unclassified | 0.443841 | 1.228383 | 0.016129 |
