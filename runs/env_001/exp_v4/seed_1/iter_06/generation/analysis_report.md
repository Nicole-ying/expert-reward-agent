# Analysis Report

## Recommended Action: tune
当前骨架已运行4轮，得分稳定在-111左右，远低于目标200。best_reward与previous_reward代码完全相同，因此无需revert。主要问题是stability_penalty权重过大（-0.5 speed, -0.3 angle, -0.1 angular_vel），导致agent过于保守，progress_delta_reward系数10.0不足以驱动有效学习。soft_landing_proxy触发率极低（0.4%），几乎无贡献。建议降低stability_penalty权重（如speed系数降至-0.2），并适当提高progress_delta_reward系数（如20.0），同时考虑放宽soft_landing_proxy条件或替换为更密集的着陆奖励。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 4

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=dominant over progress signal, causing agent to be overly cautious
- soft_landing_proxy: role=proxy dir=positive issue=too sparse (0.4% trigger rate) to provide useful learning signal
- energy_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
