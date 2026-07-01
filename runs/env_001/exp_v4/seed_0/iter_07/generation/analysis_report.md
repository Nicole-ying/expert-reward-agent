# Analysis Report

## Recommended Action: revert
当前得分-118.81远低于best得分-18.56（差距>100）。对比代码：best使用progress_scale=80.0，当前仅10.0；best的stability_penalty权重总和约0.035，当前约0.5（speed_scale=0.3, angle_scale=0.2, angular_vel_scale=0.1乘以stability_scale=0.5），导致stability_penalty主导；best的soft_landing_proxy使用连续塑形（多级奖励），当前仅单一阈值2.0且触发率极低。建议恢复到best配置，仅做小幅调整。

## Skeleton Status
- family: progress+stability+landing_proxy+energy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_reward: role=progress dir=positive issue=progress_scale=10.0 is too low compared to best's 80.0, resulting in weak learning signal
- stability_penalty: role=constraint dir=negative issue=stability_scale=0.5 with speed_scale=0.3, angle_scale=0.2, angular_vel_scale=0.1 produces penalty magnitude ~0.3*speed+0.2*angle+0.1*angular_vel, which dominates total reward (mean -0.167 vs progress mean 0.162)
- soft_landing_bonus: role=proxy dir=positive issue=trigger rate only 0.004, too low to provide useful learning signal; best version had continuous shaping with multiple tiers
- energy_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: stability_penalty_dominance, goal_near_oscillation
- hacking_risks: stability_penalty_dominance
