# Analysis Report

## Recommended Action: tune
当前骨架已迭代6轮，最佳得分-18.56，远低于目标200，且最近两轮得分重复-18.56和-48.55，陷入停滞。best_reward.py与previous_reward.py代码相同，无需revert。soft_landing_proxy触发率低（0.21），建议放宽阈值或增加权重以提升学习信号。progress_delta_reward系数80.0有效，可保持。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 6

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=none
- soft_landing_proxy: role=proxy dir=positive issue=trigger rate low (0.21), may need threshold relaxation or weight increase
- action_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, contact_reward_hacking
- hacking_risks: contact_reward_hacking
