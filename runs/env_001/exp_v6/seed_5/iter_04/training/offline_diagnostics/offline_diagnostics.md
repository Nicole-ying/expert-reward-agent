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
| original_return | 147.514617 | 21.535218 | 147.791414 | 116.343721 | 183.580521 |
| generated_return | 723.234494 | 127.314483 | 759.065636 | 234.672534 | 794.855973 |
| episode_length | 1000.000000 | 0.000000 | 1000.000000 | 1000.000000 | 1000.000000 |

- termination_rate: 0.000000
- truncation_rate: 1.000000

## Additive-Term Inference

- status: exact
- reward_terms: landing_proxy, progress_reward, stability_penalty
- diagnostics/modulators: dist_gate
- mean_abs_residual: 0.000000

## Step Component Statistics

| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dist_gate | 0.738046 | 0.738046 | 1.000000 | 0.738046 | 0.738046 | 0.119641 | 0.995229 | 20000 |
| generated_reward | 0.723234 | 0.723340 | 1.000000 | 0.723234 | 0.723340 | -0.038572 | 0.989458 | 20000 |
| landing_proxy | 0.716749 | 0.716749 | 0.776100 | 0.923527 | 0.923527 | 0.000000 | 0.989849 | 20000 |
| progress_reward | 0.006848 | 0.007206 | 1.000000 | 0.006848 | 0.007206 | -0.038119 | 0.081201 | 20000 |
| stability_penalty | -0.000363 | 0.000363 | 1.000000 | -0.000363 | 0.000363 | -0.005732 | -0.000002 | 20000 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| dist_gate | 738.045911 | 52.666780 | 754.598397 | 570.029862 | 794.845856 | 20 |
| landing_proxy | 716.749039 | 127.216796 | 752.580199 | 228.507681 | 788.401843 | 20 |
| progress_reward | 6.848421 | 0.102439 | 6.871225 | 6.494932 | 6.953108 | 20 |
| stability_penalty | -0.362966 | 0.069869 | -0.359020 | -0.496232 | -0.267749 | 20 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10005 | 123.219687 | 738.466743 | 1000 | False | True | timeout | 0.039505 | 0.004127 | 0.777000 |
| 1 | 10006 | 158.446172 | 774.901618 | 1000 | False | True | timeout | 0.026653 | 0.003303 | 0.795000 |
| 2 | 10007 | 135.386503 | 234.672534 | 1000 | False | True | timeout | 0.087482 | 0.013047 | 0.042000 |
| 3 | 10008 | 120.542448 | 750.527155 | 1000 | False | True | timeout | 0.029802 | 0.003532 | 0.768000 |
| 4 | 10009 | 167.971586 | 774.152278 | 1000 | False | True | timeout | 0.030848 | 0.003659 | 0.800000 |
| 5 | 10010 | 183.580521 | 791.130247 | 1000 | False | True | timeout | 0.031881 | 0.003070 | 0.815000 |
| 6 | 10011 | 167.795909 | 783.991700 | 1000 | False | True | timeout | 0.033196 | 0.002945 | 0.806000 |
| 7 | 10012 | 162.470880 | 771.320997 | 1000 | False | True | timeout | 0.031714 | 0.004137 | 0.803000 |
| 8 | 10013 | 173.017237 | 750.613751 | 1000 | False | True | timeout | 0.037473 | 0.003496 | 0.780000 |
| 9 | 10014 | 143.631844 | 774.952295 | 1000 | False | True | timeout | 0.029119 | 0.003577 | 0.803000 |
| 10 | 10015 | 125.866501 | 739.543150 | 1000 | False | True | timeout | 0.038641 | 0.005381 | 0.774000 |
| 11 | 10016 | 118.161109 | 760.030764 | 1000 | False | True | timeout | 0.042946 | 0.002872 | 0.799000 |
| 12 | 10017 | 155.016530 | 750.581450 | 1000 | False | True | timeout | 0.039440 | 0.003815 | 0.787000 |
| 13 | 10018 | 142.713491 | 769.983248 | 1000 | False | True | timeout | 0.037268 | 0.003751 | 0.800000 |
| 14 | 10019 | 123.046145 | 764.453617 | 1000 | False | True | timeout | 0.031004 | 0.002659 | 0.793000 |
| 15 | 10020 | 151.950985 | 495.672643 | 1000 | False | True | timeout | 0.112708 | 0.003160 | 0.346000 |
| 16 | 10021 | 177.506817 | 794.855973 | 1000 | False | True | timeout | 0.031339 | 0.003696 | 0.816000 |
| 17 | 10022 | 132.639454 | 740.451737 | 1000 | False | True | timeout | 0.028493 | 0.003950 | 0.768000 |
| 18 | 10023 | 116.343721 | 746.287478 | 1000 | False | True | timeout | 0.034725 | 0.002930 | 0.774000 |
| 19 | 10024 | 170.984806 | 758.100507 | 1000 | False | True | timeout | 0.032069 | 0.002784 | 0.779000 |
