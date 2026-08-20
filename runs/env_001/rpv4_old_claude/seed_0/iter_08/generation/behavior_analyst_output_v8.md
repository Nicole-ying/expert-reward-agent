# 训练行为诊断报告

## 1. 当前策略行为画像
策略学会了"悬停拖延"：全部 20 个 episode 跑满 1000 步都未着陆（terminated=0/20），velocity_brake 每步惩罚 -0.023，orientation_stability 每步 -0.004，而 contact_landing_reward 完全未激活（0.0%）。策略在空中维持稳定姿态但不接触地面。

## 2. 组件健康度
- **goal_approach_progress**: 健康（100% active, +0.011/step，提供持续接近奖励）
- **velocity_brake_near_target**: 主导但错位（-0.023/step 持续惩罚，占 58.1% magnitude，迫使策略远离地面以避免 proximity_factor 放大）
- **orientation_stability**: 健康（-0.004/step 小幅约束角度）
- **contact_landing_reward**: **死组件**（0.0% active，质量门槛过严：需同时满足双腿接触 + velocity<0.4 + angle<0.3 + distance<0.6）

## 3. 骨架演进表

```
| iter | score   | len    | term | contact_ep | contact_act | velocity_ep | progress_ep | orient_ep |
|------|---------|--------|------|------------|-------------|-------------|-------------|-----------|
| 2    | +142.32 | 721.1  | 11/20| +1516      | 50.3%       | -126        | +19.5       | -44.7     |
| 6    | +17.87  | 784.4  | 11/20| +1649      | 39.3%       | -158        | +23.5       | -50.2     |
| 7    | -31.83  | 1000.0 | 0/20 | **0**      | **0.0%**    | -228        | +10.7       | -37.0     |
```

**关键变化（iter 6→7）**：contact_landing_reward 从连续奖励退化为严格质量门（iter7 代码 L50-55），导致激活率从 39.3% 崩溃至 0.0%。velocity_brake 惩罚因更长 episode 累积增大（-158→-228），策略为避免惩罚选择悬停而非着陆。

**交互效应**：contact 门槛过严 → 策略无法获得着陆引导 → 为降低 velocity_brake 损失而远离地面 → len 跑满但 score 转负。

## 4. 行为结论
**病态**。策略陷入"悬停陷阱"：contact_landing_reward 的严格质量门让着陆尝试得不到任何正反馈，velocity_brake_near_target 的 proximity_factor 设计反而惩罚接近目标，两者共同导致策略学会"空中稳定盘旋"而非着陆。

**最需干预组件**：
1. **contact_landing_reward（致命）**：从 iter2 历史最佳的连续质量调制（contact_active * quality_factor^0.333）退化为全有全无门槛，直接杀死唯一的着陆引导信号。
2. **velocity_brake_near_target（结构性错误）**：proximity_factor 设计在接近目标时放大速度惩罚，与任务目标相悖——应该鼓励慢速接近而非惩罚接近本身。