# Analysis Report

## Recommended Action: revert
Best score 67.07 achieved with progress_reward=50, stability_penalty lighter (angle=0.2, angular=0.1, speed=0.15), landing_shaping coefficient=3.0 and relaxed thresholds (dist=1.0, speed=1.0, angle=0.5). Current version increased progress to 80, tightened landing thresholds (dist=0.6, speed=0.5, angle=0.3) and reduced coefficient to 1.5, and slightly increased stability penalties. Score dropped to -113.70. The tightening of landing shaping made it too sparse (nonzero rate 0.0077), and the increased progress coefficient may cause oscillation. Revert to best coefficients and only make small adjustments.

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: True
- iterations_on_skeleton: 3

## Component Analysis
- progress_reward: role=progress dir=positive issue=coefficient increased from 50 to 80, but score dropped; may cause oscillation near goal
- stability_penalty: role=constraint dir=negative issue=weights increased slightly, but mean magnitude 0.238 is above recommended 0.2; may hinder movement
- landing_shaping: role=proxy dir=positive issue=nonzero rate only 0.0077; conditions too tight (thresholds reduced), coefficient halved to 1.5; too sparse to provide useful signal

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: contact_reward_hacking
