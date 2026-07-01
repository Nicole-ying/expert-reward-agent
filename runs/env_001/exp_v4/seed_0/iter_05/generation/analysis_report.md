# Analysis Report

## Recommended Action: tune
当前骨架已迭代5轮，最佳得分-18.56远低于目标200，且最近两轮无改善。current与best代码相同，得分一致，无需revert。soft_landing_proxy触发率低（0.21），建议放宽阈值或增加权重以提升学习信号。progress_delta_reward信号强，保持。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 5

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=none
- soft_landing_proxy: role=proxy dir=positive issue=trigger rate low (0.21), may need threshold relaxation or weight increase
- action_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, contact_reward_hacking
- hacking_risks: contact_reward_hacking
