# Analysis Report

## Recommended Action: tune
stability_penalty dominates (mean -0.340, nonzero 1.0) causing early failure/crash; progress_reward too weak (mean 0.160) to counteract; soft_landing_bonus too sparse (0.5% trigger). Recommend reducing stability_penalty coefficient and increasing progress_reward coefficient.

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=mean 0.160, but external score is very negative; progress alone insufficient to overcome penalties
- stability_penalty: role=constraint dir=negative issue=mean -0.340, nonzero_rate 1.0, dominates total reward; agent penalized for moving
- soft_landing_bonus: role=proxy dir=positive issue=nonzero_rate 0.005, rarely triggered; too sparse to guide learning

## Detected Issues
- failure_modes: stability_penalty_dominance, early_failure_or_crash
- hacking_risks: stability_penalty_dominance
