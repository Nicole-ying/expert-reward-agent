# 训练行为诊断报告

## 1. 当前策略行为画像
策略学会了接近目标并维持接触（contact_landing_reward 累计 242.5，active_rate 6.3%），但回合长度暴涨至 930 步且 terminated 率从 80% 跌至 35%，说明策略陷入"贴地徘徊"——触碰目标但无法满足终止条件。

## 2. 组件健康度
- **contact_landing_reward**: 主导但错位（占 65.1% share，但仅 6.3% active 说明只在局部触发，per-step=0.26 远超其他组件）
- **goal_approach_progress**: 淹没（active_rate 99.6% 全覆盖但 per-step=0.06，被 contact 量级压制 4 倍）
- **velocity_brake_near_target**: 健康（全覆盖惩罚，per-step=-0.048，合理约束速度）
- **orientation_stability**: 健康（全覆盖惩罚，per-step=-0.007，量级适中）

## 3. 骨架演进表

```
| iter | score   | len    | term | contact_ep | contact_act | progress_ep | progress_act | velocity_ep | orient_ep |
|------|---------|--------|------|------------|-------------|-------------|--------------|-------------|-----------|
| 2    | 142.32  | 721.1  | 16/20| 242.5      | 50.8%       | 57.1        | 100.0%       | -44.5       | -6.7      |
| 3    | 36.97   | 930.1  | 7/20 | 242.5      | 6.3%        | 57.1        | 99.6%        | -44.5       | -6.7      |
```

**关键发现**：iter 2→3 唯一修改是将 `goal_approach_progress` 系数从 10.0 提升至 50.0，但：
- **期望效果**：增强导航信号，加速接近目标
- **实际连锁反应**：
  - contact_landing_reward 的 **active_rate 从 50.8% 暴跌至 6.3%**（降低 9 倍），说明策略改为"快速冲刺→短暂接触→弹开"而非持续接触
  - 回合长度增加 209 步，terminated 减少 9 个，证实策略无法稳定着陆
  - 两个组件的 episode_sum 完全相同，说明**修改未生效**或被其他因素抵消

**交互效应诊断**：提升 progress 系数反而破坏了 iter 2 的稳定接触行为。可能原因：
1. progress 信号鼓励"快速逼近"与 contact 要求的"缓慢稳定接触"冲突
2. contact 组件的 quality_factor 三项乘积设计过于苛刻，高速接近时质量因子崩溃为零

## 4. 行为结论
**病态**。策略从 iter 2 的"稳定接触但未达终止条件"退化为 iter 3 的"快速冲刺但无法维持接触"。

**最需要干预的组件**：
1. **contact_landing_reward**（首要）：active_rate 6.3% 说明触发窗口过窄，三项质量因子的乘积设计导致稍有偏差即归零。需要放宽质量阈值或改用加权和而非乘积。
2. **goal_approach_progress**（次要）：当前 per-step 量级被 contact 压制，且"逼近速度"与"着陆质量"存在根本冲突。需要设计分阶段衰减或距离自适应缩放。