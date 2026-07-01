# Analysis Report

## Recommended Action: tune
Current score -101.17 is best so far, but still far from target 200. Progress_reward coefficient 10.0 is weak; stability_penalty dominates. Landing_shaping rarely activates. Tune: increase progress_reward coefficient to 50.0, reduce stability_penalty weights, and relax landing_shaping conditions to increase activation.

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient 10.0 may be too low to drive sufficient progress; mean 0.16 indicates weak positive signal.
- stability_penalty: role=constraint dir=negative issue=Mean -0.34 dominates total reward; may be too aggressive, causing agent to move slowly.
- landing_shaping: role=proxy dir=positive issue=Very low nonzero rate (0.0077) and mean 0.013; rarely activates, likely due to strict conditions.

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: contact_reward_hacking
