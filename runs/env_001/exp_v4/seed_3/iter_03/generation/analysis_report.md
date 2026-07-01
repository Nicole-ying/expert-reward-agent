# Analysis Report

## Recommended Action: tune
当前骨架已运行3轮，得分稳定在-110左右，无显著改善。best_reward.py与previous_reward.py几乎相同（仅landing_bonus从1.0提升到1.5），但best得分-111.98略低于当前-110.51，因此无需revert。主要问题是stability_penalty系数过高（0.5,0.3,0.2）导致其主导总奖励，建议降低系数（如0.2,0.1,0.1）以鼓励探索；同时progress_reward系数2.0可能过低，可尝试提升至5.0或10.0以增强学习信号；landing_bonus条件可适当放宽（如dist<0.8, speed<0.5, angle<0.3）以提高触发率。

## Skeleton Status
- family: progress+stability+landing_proxy+energy
- stagnant: True
- iterations_on_skeleton: 3

## Component Analysis
- progress_reward: role=progress dir=positive issue=coefficient 2.0 may be too low to drive meaningful progress; mean reward is only 0.032
- stability_penalty: role=constraint dir=negative issue=dominates total reward (mean -0.242 vs total -0.214), likely causing conservative behavior
- landing_bonus: role=proxy dir=positive issue=nonzero rate extremely low (0.0055), indicating landing conditions are too strict or rarely achieved
- energy_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
