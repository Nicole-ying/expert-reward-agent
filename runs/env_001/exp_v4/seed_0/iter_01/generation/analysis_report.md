# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-111.31远低于目标200。best_reward与previous_reward代码完全相同，因此无revert必要。主要问题：stability_penalty系数过高（angle_penalty=0.5, angular_vel_penalty=0.3, speed_penalty=0.2）导致惩罚主导，progress_reward系数10.0相对不足。建议降低stability_penalty系数（如angle_penalty=0.2, angular_vel_penalty=0.1, speed_penalty=0.1）并提高progress_reward系数（如20.0），以平衡驱动与稳定。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=Coefficient 10.0 may be too low relative to stability penalty, leading to weak drive.
- stability_penalty: role=constraint dir=negative issue=Mean -0.241 with abs_mean 0.241 dominates total reward (mean -0.078). Penalty coefficients (0.5, 0.3, 0.2) may be too high, causing agent to be overly cautious.
- soft_landing_bonus: role=proxy dir=positive issue=Nonzero rate only 0.5%, too sparse to provide meaningful guidance.
- energy_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
