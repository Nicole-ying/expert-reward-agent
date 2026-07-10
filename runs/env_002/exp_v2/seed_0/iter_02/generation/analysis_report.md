# Analysis Report

## Recommended Action: tune
当前得分11.95远低于目标200，但相比上一轮-102.55有显著提升。best_reward与previous_reward代码完全相同，无需revert。骨架仅迭代2轮，尚未停滞。主要问题：progress_delta_reward系数5.0可能不足，建议提升至10-20；stability_penalty过弱，建议增大角度和角速度惩罚系数；soft_landing_bonus稀疏且条件宽松，可能引发contact_reward_hacking，建议收紧条件或降低权重。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=mean positive but abs_mean 0.08 suggests oscillation; may cause goal_near_oscillation
- stability_penalty: role=constraint dir=negative issue=nonzero_rate 0.26, mean -0.004; too weak to prevent instability
- energy_penalty: role=efficiency dir=negative issue=none
- soft_landing_bonus: role=proxy dir=positive issue=nonzero_rate 0.146, mean 0.073; sparse, may cause contact_reward_hacking

## Detected Issues
- failure_modes: high_reward_without_success, goal_near_oscillation
- hacking_risks: contact_reward_hacking
