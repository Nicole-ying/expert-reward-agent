# Analysis Report

## Recommended Action: revert
Best score (-108.58) is higher than current (-112.24). The main change was increasing progress_reward coefficient from 10 to 50 and adding distance_anchor. The increase likely caused oscillation (goal_near_oscillation) and the anchor added negative bias. Revert to best_reward.py coefficients (progress=10, no distance_anchor) and only tune stability_penalty or landing_shaping slightly.

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient increased from 10 to 50, but score dropped. May be too aggressive, causing oscillation near goal.
- distance_anchor: role=anchor dir=negative issue=Added new component with -0.5 weight. Negative mean indicates it penalizes distance, but may conflict with progress_reward.
- stability_penalty: role=constraint dir=negative issue=Conditional penalty near target may be too strong, hindering settling.
- landing_shaping: role=proxy dir=positive issue=Very low nonzero_rate (0.007), almost never activated. Ineffective.

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: goal_near_oscillation
