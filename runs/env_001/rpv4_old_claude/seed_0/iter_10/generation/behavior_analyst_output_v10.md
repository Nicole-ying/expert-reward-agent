## 1. 当前策略行为画像
策略学会了接触地面并保持较短回合（len=755），60% episode 能触发终止（12/20 terminated），但得分剧烈波动（range [-132.8, 181.4]）且均值转负（-1.59），说明**接触时机和质量不稳定，部分 episode 是高速撞毁**。

## 2. 组件健康度
- **contact_landing_reward**: 主导但错位（72.4% share，但 active_rate 仅 6.9%，说明接触集中在末尾少数步，且 per-step=140.04/755=0.185，量级远超其他组件）
- **goal_approach_progress**: 淹没（4.5% share，per-step=8.65/755=0.011，被 contact 信号遮蔽 17 倍）
- **velocity_brake_near_target**: 全覆盖惩罚（99.8% active，per-step=-0.045，持续负反馈但量级弱）
- **orientation_stability**: 全覆盖惩罚（100% active，per-step=-0.007，几乎无影响）

## 3. 骨架演进表

```
| iter | score   | len    | term | contact_ep | contact_active | progress_ep | progress_active | velocity_ep | orient_ep |
|------|---------|--------|------|------------|----------------|-------------|-----------------|-------------|-----------|
| 2    | 142.32  | 721.10 | ?/20 | ?          | ?              | ?           | ?               | ?           | ?         |
| 8    | 8.81    | 848.00 | 9/20 | 2.441*848  | 49.5%          | 0.030*848   | 100.0%          | -0.205*848  | -0.071*848|
| 9    | -1.59   | 755.05 | 12/20| 140.04     | 6.9%           | 8.65        | 99.7%           | -34.26      | -5.33     |
```

**iter 8→9 修改**: 代码完全相同，但统计口径不同。iter 8 用 per-step 均值（训练分布），iter 9 用 episode_sum_mean（固定评估集）。真正变化是：
- contact 从"稠密但弱"（49.5% active，per-step 2.44）变成"稀疏但强"（6.9% active，episode 140），说明策略从"频繁低质量接触"转向"集中末尾高质量接触"。
- 连锁反应：len 缩短 93 步，term 增加 3 个，但 score 暴跌 10.4，说明**接触质量因子失效**——策略找到了"快速撞地触发 contact 但 quality_factor 崩塌"的捷径。

## 4. 行为结论
**病态**。策略已收敛到局部解：用高速撞地触发 contact_active（满足 max(left, right)=1.0），但 velocity/angle/proximity quality 三者相乘后接近 0，导致 contact reward 名义值高但实际贡献崩塌。得分波动 [-132, 181] 说明部分种子侥幸成功，大部分在撞毁。

**最需要干预的组件**: **contact_landing_reward**  
**原因**: 当前设计"contact_active * quality_factor"允许策略先满足 contact（简单），再忽略 quality（困难）。quality 三因子相乘（0.333 次方缓解但不够）在极端情况下趋零，使 contact 信号退化为"是否接地"的二值开关，而非"优质着陆"的引导。需要重构为"无接触零奖励，接触后 quality 线性/凸组合加成"，或拆分为多个独立稀疏事件（如 safe_contact_bonus 需同时满足速度<阈值 & 角度<阈值 & 距离<阈值）。