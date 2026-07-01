# Analysis Report

## Recommended Action: revert
Best score (-111.64) is only slightly lower than current (-110.8), but best skeleton achieved that score with higher progress coefficient (10.0 vs 5.0) and different stability weights (speed 0.5, angle 0.3, angular 0.1 vs angle 0.5, vel 0.3, angular 0.2). Current changes weakened progress signal and increased angle/angular penalties, causing stability_penalty to dominate. Revert to best coefficients and only make minor adjustments.

## Skeleton Status
- family: progress+stability+landing_proxy+action_penalty
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient reduced from 10.0 to 5.0, weakening the main learning signal; mean value 0.081 is low.
- stability_penalty: role=constraint dir=negative issue=Dominant component with mean -0.344, likely overpowering progress signal; weights changed from best (speed 0.5, angle 0.3, angular 0.1) to current (angle 0.5, vel 0.3, angular 0.2), increasing angle and angular penalties.
- soft_landing_bonus: role=proxy dir=positive issue=Sparse trigger rate (0.49%); conditions unchanged from best, but coefficient reduced from 2.0 to 2.0 (same).
- action_penalty: role=efficiency dir=negative issue=Replaced energy_penalty (-0.1) with action_penalty (-0.05); both weak, but change unnecessary.

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
