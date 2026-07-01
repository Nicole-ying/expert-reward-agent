# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-111.64远低于目标200。best_reward与previous_reward完全相同，无需revert。stability_penalty均值-0.549，绝对值大于progress_delta_reward均值0.161，导致总奖励为负，agent可能过于保守。建议降低stability_penalty系数（如speed系数从0.5降至0.2），同时提高progress_delta_reward系数（如从10.0增至20.0）以增强接近目标的驱动力。soft_landing_proxy触发率极低，可放宽条件（如near_target阈值从0.3增至0.5）或暂时移除。

## Skeleton Status
- family: progress+stability+landing_proxy+energy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=dominant negative signal (mean -0.549) may suppress exploration
- soft_landing_proxy: role=proxy dir=positive issue=very sparse (nonzero_rate 0.004), may be too strict
- energy_penalty: role=efficiency dir=negative issue=none

## Detected Issues
- failure_modes: goal_near_oscillation, stability_penalty_dominance
- hacking_risks: stability_penalty_dominance
