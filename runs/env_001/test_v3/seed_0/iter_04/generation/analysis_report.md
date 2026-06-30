# Analysis Report

## Recommended Action: tune
本轮外部得分-142.80，远低于目标200。progress_reward均值1.599，信号强但不足以克服stability_penalty和原始环境惩罚。landing_shaping触发率仅1%，稀疏无效。历史中骨架迭代2次，最佳得分158.82（第3轮），但第4轮大幅下降，表明不稳定。建议：增大progress_reward系数至200以上，降低stability_penalty系数至0.05，提高landing_shaping系数至5.0并放宽条件以增加触发率。

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: False
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=none
- stability_penalty: role=constraint dir=negative issue=penalty dominance: generated_reward mean 1.488 vs stability_penalty mean -0.122, but nonzero_rate 1.0 indicates constant penalty; coefficient 0.1 may be too high
- landing_shaping: role=proxy dir=positive issue=sparse proxy: nonzero_rate 0.010386, too low to guide learning

## Detected Issues
- failure_modes: early_failure_or_crash, goal_near_oscillation
- hacking_risks: stability_penalty_dominance
