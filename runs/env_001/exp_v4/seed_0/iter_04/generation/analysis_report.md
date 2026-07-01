# Analysis Report

## Recommended Action: revert
当前得分-48.55远低于best得分-18.56。对比代码，current将soft_landing_proxy阈值从0.3/0.3/0.2放宽到0.5/0.5/0.3，权重从3.0提升到5.0，导致contact_reward_hacking（触发率0.45但得分下降）。best的收紧阈值和较低权重更有效。建议revert到best配置，仅微调。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 4

## Component Analysis
- action_penalty: role=constraint dir=negative issue=none
- angle_penalty: role=constraint dir=negative issue=none
- angular_vel_penalty: role=constraint dir=negative issue=none
- progress_delta_reward: role=progress dir=positive issue=none
- soft_landing_proxy: role=proxy dir=positive issue=thresholds too loose (0.5/0.5/0.3) causing contact_reward_hacking; weight increased to 5.0 may over-reward hacking
- speed_penalty: role=constraint dir=negative issue=none
- stability_penalty: role=constraint dir=negative issue=none

## Detected Issues
- failure_modes: contact_reward_hacking, goal_near_oscillation
- hacking_risks: contact_reward_hacking
