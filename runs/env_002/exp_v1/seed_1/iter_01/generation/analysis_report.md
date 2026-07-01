# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-102.17远低于目标200。stability_penalty均值-0.134远大于progress_delta_reward均值-0.005，导致agent不敢移动。best_reward与current代码完全相同，因此无需revert。建议降低stability_penalty系数（如将angle_penalty从-1.0降至-0.5），并提高progress_delta_reward系数（如从2.0增至5.0），以平衡探索与稳定。

## Skeleton Status
- family: progress+stability+energy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=mean near zero, suggesting insufficient drive; coefficient 2.0 may be too low
- stability_penalty: role=constraint dir=negative issue=dominates total reward (mean -0.134 vs progress -0.005), causing agent to be overly conservative
- energy_penalty: role=efficiency dir=negative issue=small magnitude, acceptable

## Detected Issues
- failure_modes: stability_penalty_dominance, generated_reward_negative_average
- hacking_risks: stability_penalty_dominance
