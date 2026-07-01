# Analysis Report

## Recommended Action: tune
当前得分-27.09，较上一轮-119.87大幅提升，但距离目标200仍远。best_reward与previous_reward代码完全相同，无需revert。soft_landing_proxy触发率60.4%但mean 0.957，可能条件过宽导致contact_reward_hacking。progress_delta_reward信号偏弱（mean 0.03），建议增大progress_scale至50-100。stability_penalty权重已较低，暂不调整。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=mean 0.0306, abs_mean 0.0339, but progress_scale=5.0 may be too low to drive strong learning; consider increasing to 50-100
- stability_penalty: role=constraint dir=negative issue=weights are low (speed 0.02, angle 0.01, angular_vel 0.005), mean -0.0099, not dominating
- soft_landing_proxy: role=proxy dir=positive issue=mean 0.957, nonzero_rate 0.604, but conditions may be too loose (thresholds 0.5, 0.5, 0.3) leading to contact_reward_hacking; high reward without true success
- action_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: contact_reward_hacking
