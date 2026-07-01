# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-119.87远低于目标200。best_reward与previous_reward完全相同，无需revert。主要问题是stability_penalty和speed_penalty权重过大（-0.1086和-0.1056），压制了progress_delta_reward的正信号（0.054）。soft_landing_proxy触发率极低（0.0043），条件过严。建议降低speed_penalty_weight和angle_penalty_weight，放宽soft_landing_proxy条件，并增加progress_scale以增强学习信号。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=mean=0.054, but abs_mean=0.057 indicates oscillation; may cause goal_near_oscillation
- stability_penalty: role=constraint dir=negative issue=mean=-0.1086, abs_mean=0.1086, nonzero_rate=1.0; dominates total reward, likely causing stability_penalty_dominance
- speed_penalty: role=constraint dir=negative issue=mean=-0.1056, abs_mean=0.1056; contributes to stability_penalty_dominance
- angle_penalty: role=constraint dir=negative issue=mean=-0.0016, abs_mean=0.0016; minor
- angular_vel_penalty: role=constraint dir=negative issue=mean=-0.0014, abs_mean=0.0014; minor
- soft_landing_proxy: role=proxy dir=positive issue=nonzero_rate=0.0043, mean=0.0043; rarely triggered, may be too strict
- action_penalty: role=efficiency dir=negative issue=mean=-0.002, nonzero_rate=0.197; negligible

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
