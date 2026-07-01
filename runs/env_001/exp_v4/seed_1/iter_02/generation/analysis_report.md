# Analysis Report

## Recommended Action: revert
Best score (-111.64) is higher than current (-120.28) by 8.64. The only changes from best were: progress_delta_reward coefficient 10->20, stability_penalty weights reduced (speed 0.5->0.2, angle 0.3->0.1, angular_vel 0.1->0.05), soft_landing_proxy conditions relaxed (dist 0.3->0.5, speed 0.2->0.3, angle 0.2->0.3) and reward 2.0->3.0. These changes caused regression. Revert to best coefficients and only make small adjustments.

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=Coefficient increased from 10 to 20, but score dropped. May cause overshooting or oscillation near goal.
- stability_penalty: role=constraint dir=negative issue=Weights reduced (speed 0.5->0.2, angle 0.3->0.1, angular_vel 0.1->0.05). Mean penalty decreased from -0.549 to -0.222, but score worsened. Possibly too weak to prevent instability.
- soft_landing_proxy: role=proxy dir=positive issue=Conditions relaxed (dist<0.5, speed<0.3, angle<0.3) and reward increased to 3.0, but nonzero_rate remains very low (0.0049). Too sparse to provide useful signal.
- energy_penalty: role=efficiency dir=negative issue=Unchanged from best. Minor impact.

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: goal_near_oscillation
