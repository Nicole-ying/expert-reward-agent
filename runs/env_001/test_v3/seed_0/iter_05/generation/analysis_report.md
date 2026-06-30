# Analysis Report

## Recommended Action: rebuild
当前骨架已迭代3轮，最佳得分158.82远低于目标200，且最近两轮得分大幅下降（-142.80, -190.67）。landing_shaping触发率极低（0.2%），无法提供有效学习信号；stability_penalty始终为负且全触发，可能抑制探索。progress_reward均值5.12但外部评分-190.67，说明progress_reward与外部目标不一致。建议重建骨架，移除landing_shaping，大幅提高progress_reward系数（>200），降低stability_penalty系数（<0.1），并考虑加入其他引导信号。

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: True
- iterations_on_skeleton: 3

## Component Analysis
- progress_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=penalty dominance: generated_reward mean 5.06 but stability_penalty mean -0.067, nonzero_rate 1.0, may still be too high relative to progress
- landing_shaping: role=proxy dir=positive issue=sparse proxy: nonzero_rate 0.002, mean 0.01, not providing useful signal

## Detected Issues
- failure_modes: early_failure_or_crash, goal_near_oscillation
- hacking_risks: stability_penalty_dominance
