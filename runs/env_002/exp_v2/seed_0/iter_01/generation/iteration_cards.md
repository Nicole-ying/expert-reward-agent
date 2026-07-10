## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## generated_reward_negative_average
- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions
