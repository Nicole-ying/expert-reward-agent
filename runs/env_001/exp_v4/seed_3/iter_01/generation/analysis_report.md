# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-111.98远低于目标200。best_reward与previous_reward完全相同，因此无需revert。主要问题是progress_reward系数过低（2.0），导致学习信号弱，而stability_penalty系数相对过大，主导了总奖励。建议增大progress_scale（如10.0），同时降低stability_penalty各系数（如angle_penalty=0.2, angular_vel_penalty=0.1, speed_penalty=0.1），并考虑提高landing_bonus或使其更易触发。

## Skeleton Status
- family: progress+stability+landing_proxy+energy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=progress_scale=2.0 too low; mean reward only 0.032, insufficient to drive learning
- stability_penalty: role=constraint dir=negative issue=dominates total reward (mean -0.243 vs progress 0.032), causing agent to be overly conservative
- landing_bonus: role=proxy dir=positive issue=extremely sparse (nonzero_rate=0.005), provides almost no learning signal
- energy_penalty: role=efficiency dir=negative issue=small magnitude, but may discourage movement when combined with stability penalty

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
