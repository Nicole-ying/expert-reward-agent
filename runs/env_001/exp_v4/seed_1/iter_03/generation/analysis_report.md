# Analysis Report

## Recommended Action: revert
Best score (-111.64) is higher than current (-115.99). The only change from best was reducing angle_penalty weight from 0.3 to 0.25 and increasing soft_landing_proxy from 2.0 to 2.5. The reduction in stability penalty likely increased oscillation (goal_near_oscillation) without improving landing success. Revert to best coefficients (angle_penalty=0.3, soft_landing_proxy=2.0) and consider increasing progress_delta_reward coefficient to better counteract stability penalty.

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 3

## Component Analysis
- energy_penalty: role=constraint dir=negative issue=none
- progress_delta_reward: role=progress dir=positive issue=coefficient 10.0 may be insufficient to overcome stability penalty
- soft_landing_proxy: role=proxy dir=positive issue=sparse trigger rate (0.4%) despite reward increase to 2.5
- stability_penalty: role=constraint dir=negative issue=dominates total reward; mean -0.55 vs progress mean 0.16

## Detected Issues
- failure_modes: stability_penalty_dominance, goal_near_oscillation
- hacking_risks: stability_penalty_dominance
