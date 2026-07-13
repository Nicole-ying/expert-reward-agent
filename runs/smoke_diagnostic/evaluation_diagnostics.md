# Offline Policy Diagnostics

> Shadow-mode replay only. These statistics describe deterministic evaluation trajectories, not historical training trajectories.

## Replay

- env_id: LunarLander-v2
- episodes: 2
- seed_base: 10123
- reward_errors: 0

## Evaluation Summary

| metric | mean | std | median | min | max |
|---|---:|---:|---:|---:|---:|
| original_return | -131.453043 | 1.240395 | -131.453043 | -132.693439 | -130.212648 |
| generated_return | 1.281264 | 0.489847 | 1.281264 | 0.791417 | 1.771110 |
| episode_length | 61.500000 | 0.500000 | 61.500000 | 61.000000 | 62.000000 |

- termination_rate: 1.000000
- truncation_rate: 0.000000

## Additive-Term Inference

- status: exact
- reward_terms: landing_proxy, progress_reward, stability_penalty
- diagnostics/modulators: dist_gate
- mean_abs_residual: 0.000000

## Step Component Statistics

| component | mean | abs_mean | active_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dist_gate | 0.220862 | 0.220862 | 1.000000 | 0.220862 | 0.220862 | 0.125028 | 0.564372 | 123 |
| generated_reward | 0.020834 | 0.022344 | 1.000000 | 0.020834 | 0.022344 | -0.025751 | 0.717653 | 123 |
| landing_proxy | 0.005918 | 0.005918 | 0.008130 | 0.727903 | 0.727903 | 0.000000 | 0.727903 | 123 |
| progress_reward | 0.018482 | 0.019174 | 1.000000 | 0.018482 | 0.019174 | -0.011992 | 0.032878 | 123 |
| stability_penalty | -0.003567 | 0.003567 | 1.000000 | -0.003567 | 0.003567 | -0.013759 | -0.000502 | 123 |

## Per-Episode Component Sums

| component | mean | std | median | min | max | episodes |
|---|---:|---:|---:|---:|---:|---:|
| dist_gate | 13.583035 | 0.630710 | 13.583035 | 12.952325 | 14.213745 | 2 |
| landing_proxy | 0.363951 | 0.363951 | 0.363951 | 0.000000 | 0.727903 | 2 |
| progress_reward | 1.136661 | 0.104494 | 1.136661 | 1.032167 | 1.241155 | 2 |
| stability_penalty | -0.219349 | 0.021401 | -0.219349 | -0.240750 | -0.197948 | 2 |

## Episodes

| episode | seed | original_return | generated_return | length | terminated | truncated | outcome | final_distance | final_speed | both_contact_rate |
|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 10123 | -132.693439 | 1.771110 | 61 | True | False | terminated_unclassified | 0.164410 | 0.128250 | 0.016393 |
| 1 | 10124 | -130.212648 | 0.791417 | 62 | True | False | terminated_unclassified | 0.373169 | 1.357492 | 0.000000 |
