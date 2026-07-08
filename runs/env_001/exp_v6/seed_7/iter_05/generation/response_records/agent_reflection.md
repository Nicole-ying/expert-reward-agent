# Response Record

## Prediction Review

从上一轮设计说明中提取可验证的预测：

| 预测 | 判定 | 证据 |
|------|------|------|
| "智能体可能满足于在目标正上方小距离悬停" | **refuted** | distance_reward mean=-0.973，平均距离~0.97，远非"小距离悬停"；episode 全部 early terminal，说明智能体并未接近目标 |
| "time_penalty 和 energy_penalty 会持续施压推动最终接触" | **refuted** | 100% early terminal，智能体在到达目标前就终止了，惩罚压力未能推动接近行为 |
| "距离惩罚可能使智能体早期过分保守" | **inconclusive** | episode 短且 score 极负，无法判断是"保守"还是"直接失败" |

## Observed Facts

1. **Score**: -109.08, episode_length=68.3, 10/10 (100%) early_terminal (<150 steps & score<-50)
2. **distance_reward** mean=-0.973, nonzero_rate=1.0 → 平均距离约 0.97 单位，智能体始终远离目标
3. **angle_penalty** mean=-0.028, nonzero_rate=1.0 → 姿态角平均 0.028 rad (~1.6°)，姿态控制尚可
4. **angular_vel_penalty** mean=-0.067, nonzero_rate≈1.0 → 角速度约 0.067 rad/s
5. **energy_penalty** mean=-0.010, nonzero_rate=0.189 → 仅 18.9% 步使用引擎
6. **time_penalty** mean=-0.010 (恒定)，total_reward mean=-1.088/步
7. 所有 reward component 均为非正值（全惩罚），**无任何正奖励信号**
8. `original_env_reward` mean=-1.578，比 generated_reward 更负，环境本身在施加重罚
9. 历史最佳 score=-99.87（iter 1, landing_bonus+progress_delta+stability_penalty），当前 score=-109.08 略差

## Hypothesis

**根因假设**：全负奖励函数（"纯惩罚"设计）导致智能体无法区分"做得好"和"做得差"——所有行为都受罚，最优策略变成"尽快终止以最小化累积惩罚"。证据：

- 5 个 reward term 全部 ≤0，无一正值
- `distance_reward = -distance` 
