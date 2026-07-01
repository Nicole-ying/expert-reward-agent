## stability_penalty_dominance
- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy
- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping
