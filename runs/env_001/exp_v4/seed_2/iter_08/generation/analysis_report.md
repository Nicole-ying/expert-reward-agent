# Analysis Report

## Recommended Action: revert
当前得分-108.5远低于历史最佳67.07（差距175.57）。对比best_reward.py，关键差异：progress_reward系数从50降为10，导致学习信号弱；stability_penalty权重过高（angle从0.2→0.5，angular从0.1→0.3，speed从0.15→0.2），导致惩罚主导；soft_landing_bonus条件过严且系数低（2.0 vs 3.0+2.0），激活率极低。建议恢复到best_reward的系数和组件结构（使用landing_shaping替代soft_landing_bonus），仅做小幅调整。

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: True
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=coefficient too low (10.0) compared to best (50.0), resulting in weak learning signal
- stability_penalty: role=constraint dir=negative issue=weights too high (angle=-0.5, angular=-0.3, speed=-0.2) causing dominance; mean magnitude 0.23 exceeds recommended 0.2
- soft_landing_bonus: role=proxy dir=positive issue=nonzero rate only 0.0059, conditions too strict (dist<0.5, speed<0.3, angle<0.2); best uses relaxed thresholds and higher coefficient

## Detected Issues
- failure_modes: stability_penalty_dominance, sparse_completion_proxy
- hacking_risks: stability_penalty_dominance
