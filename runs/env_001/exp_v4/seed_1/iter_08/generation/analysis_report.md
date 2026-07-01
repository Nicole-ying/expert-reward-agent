# Analysis Report

## Recommended Action: revert
Current score (-110.36) is worse than best (-111.64) but best_reward.py (score -111.64) uses progress_delta_reward with coefficient 10.0 and energy_penalty -0.1, while current uses progress_reward with coefficient 5.0 and action_penalty -0.05. The best configuration achieved higher score despite similar skeleton. Revert to best_reward.py coefficients and components, then consider tuning stability_penalty weights to reduce dominance.

## Skeleton Status
- family: progress+stability+landing_proxy+action_penalty
- stagnant: True
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient too low (5.0 vs best's 10.0), resulting in weak learning signal.
- stability_penalty: role=constraint dir=negative issue=Dominates total reward (mean -0.24 vs progress 0.08), causing agent to be overly cautious.
- soft_landing_bonus: role=proxy dir=positive issue=Sparse trigger rate (0.5%) provides negligible learning signal.
- action_penalty: role=efficiency dir=negative issue=Small penalty, but may discourage necessary actions.

## Detected Issues
- failure_modes: stability_penalty_dominance, goal_near_oscillation
- hacking_risks: stability_penalty_dominance
