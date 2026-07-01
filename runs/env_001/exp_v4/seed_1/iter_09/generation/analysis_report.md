# Analysis Report

## Recommended Action: tune
当前骨架已迭代5轮，得分稳定在-114左右，未突破-111.64。best_reward.py与previous_reward.py几乎相同，仅stability_penalty系数不同：best中angle_penalty权重0.3、angular_vel_penalty权重0.1，而current中分别为0.2和0.05。best得分-111.64略高于current的-114.39，但差距不大，且best也是该骨架下的得分。主要问题是stability_penalty过强（mean -0.543 vs progress 0.162），建议小幅降低stability_penalty系数，例如将speed权重从0.5降至0.4，angle_penalty从0.2降至0.15，angular_vel_penalty从0.05降至0.03，以平衡约束与进度信号。同时soft_landing_proxy触发率极低（0.4%），可考虑放宽条件或改用连续proxy。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 5

## Component Analysis
- energy_penalty: role=constraint dir=negative issue=none
- progress_delta_reward: role=progress dir=positive issue=none
- soft_landing_proxy: role=proxy dir=positive issue=sparse trigger rate (0.4%)
- stability_penalty: role=constraint dir=negative issue=dominates total reward; mean -0.543 vs progress mean 0.162

## Detected Issues
- failure_modes: stability_penalty_dominance, sparse_completion_proxy
- hacking_risks: stability_penalty_dominance
