# Training Feedback

## Training outcome
score=242.406225, len=481.800000, errors=0

## Component evidence

| component | inferred_type | mean | abs_mean | active_rate | active_mean | episode_sum_mean | ratio_to_progress |
|---|---|---:|---:|---:|---:|---:|---:|
| dist_gate | modulator | 0.526483 | 0.526483 | 1.000000 | None | N/A | 168.653546 |
| landing_proxy | reward_term | 0.447489 | 0.447489 | 0.561646 | None | N/A | 143.348474 |
| progress_reward | reward_term | 0.003122 | 0.003423 | 0.999584 | None | N/A | 1.000000 |
| stability_penalty | reward_term | -0.001045 | 0.001045 | 1.000000 | None | N/A | -0.334648 |
| total_reward | total | 0.449566 | 0.450037 | 1.000000 | None | N/A | 144.013826 |

> `ratio_to_progress` is descriptive evidence, not a target balance. Do not change a term from this ratio alone; signed progress can cancel and event/proxy terms have different roles.

## Distribution
- score: mean=242.406225, min=109.652525, max=282.466855
- episode_length: mean=481.800000
- legacy_early_terminal_heuristic (<150 steps + score<-50): 0/10 (0%)
- terminated: 9/10
- truncated: 1/10
- errors: 0

## Behavior evidence

Observed/derived trajectory facts; exact termination reason remains unknown unless explicitly exposed.

- outcomes: terminated_unclassified=9, timeout=1
- final_distance: mean=0.066581, min=0.001419, max=0.112284
- final_speed: mean=0.000167, min=0.000000, max=0.001671
- final_abs_angle: mean=0.001783, min=0.000478, max=0.003222
- both_contact_rate: mean=0.197632, min=0.085227, max=0.707676
- action_rate_0: mean=0.116955, min=0.063000, max=0.328076
- action_rate_1: mean=0.208439, min=0.065195, max=0.418000
- action_rate_2: mean=0.439685, min=0.175605, max=0.518310
- action_rate_3: mean=0.234921, min=0.177249, max=0.431125
