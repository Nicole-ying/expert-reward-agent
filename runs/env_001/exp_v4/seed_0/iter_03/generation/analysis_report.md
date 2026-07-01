# Analysis Report

## Recommended Action: tune
当前骨架连续3轮得分上升（-119.87→-27.09→-18.56），趋势良好，但距离目标200仍有差距。best_reward与previous_reward代码完全相同，无需revert。soft_landing_proxy触发率低（0.21），建议进一步放宽条件或增加权重；progress_delta_reward信号强，可保持。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 3

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=none
- soft_landing_proxy: role=proxy dir=positive issue=trigger rate low (0.21), may need further tuning
- action_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, contact_reward_hacking
- hacking_risks: contact_reward_hacking
