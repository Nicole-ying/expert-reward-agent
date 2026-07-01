# Analysis Report

## Recommended Action: tune
Current score -108.58 is far below target 200. Progress reward coefficient (10.0) is too weak; stability penalty dominates. Best_reward.py is identical to previous, so no revert needed. Increase progress coefficient to at least 50, reduce stability penalty weights, and consider adding a distance anchor to prevent oscillation.

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient 10.0 is too low; mean progress reward is only 0.16, insufficient to drive learning toward target score 200.
- stability_penalty: role=constraint dir=negative issue=Mean penalty -0.34 dominates total reward, causing agent to be overly cautious and not make progress.
- soft_landing_bonus: role=proxy dir=positive issue=Very low nonzero rate (0.6%) and small magnitude; barely influences behavior.

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
