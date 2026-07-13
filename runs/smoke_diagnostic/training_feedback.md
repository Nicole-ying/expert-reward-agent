# Training Feedback

## Training outcome
score=-131.453043, len=61.500000, errors=0

## Component evidence

| component | inferred_type | mean | abs_mean | active_rate | active_mean | episode_sum_mean | ratio_to_progress |
|---|---|---:|---:|---:|---:|---:|---:|
| dist_gate | modulator | 0.187903 | 0.187903 | 1.000000 | 0.187903 | N/A | N/A |
| landing_proxy | reward_term | 0.000793 | 0.000793 | 0.005859 | 0.135337 | 0.077335 | 0.076185 |
| progress_reward | reward_term | 0.010409 | 0.011710 | 1.000000 | 0.010409 | 0.963570 | 1.000000 |
| stability_penalty | reward_term | -0.002898 | 0.002898 | 1.000000 | -0.002898 | -0.272128 | -0.278447 |
| total_reward | total | 0.008304 | 0.010695 | 1.000000 | 0.008304 | 0.768777 | 0.797738 |

> `ratio_to_progress` is descriptive evidence, not a target balance. Do not change a term from this ratio alone; signed progress can cancel and event/proxy terms have different roles.

## Distribution
- score: mean=-131.453043, min=-132.693439, max=-130.212648
- episode_length: mean=61.500000
- legacy_early_terminal_heuristic (<150 steps + score<-50): 2/2 (100%)
- terminated: 2/2
- truncated: 0/2
- errors: 0

## Behavior evidence

Observed/derived trajectory facts; exact termination reason remains unknown unless explicitly exposed.

- outcomes: terminated_unclassified=2
- final_distance: mean=0.268789, min=0.164410, max=0.373169
- final_speed: mean=0.742871, min=0.128250, max=1.357492
- final_abs_angle: mean=0.169544, min=0.000359, max=0.338728
- both_contact_rate: mean=0.008197, min=0.000000, max=0.016393
- action_rate_0: mean=0.919355, min=0.838710, max=1.000000
- action_rate_1: mean=0.000000, min=0.000000, max=0.000000
- action_rate_2: mean=0.000000, min=0.000000, max=0.000000
- action_rate_3: mean=0.080645, min=0.000000, max=0.161290
