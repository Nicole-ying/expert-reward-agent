# 训练数据报告（记者模式）

## 1. 行为事实
- **最终策略得分**：score = 107.13，range = [-33.72, 264.82]
- **终止方式**：terminated（成功着陆）11/20，truncated（时间上限）9/20
- **回合长度**：len = 775.60
- **Checkpoint 得分轨迹**（官方环境得分，cp1 = 10% 步数，cp10 = 100% 步数）：

  | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
  |------|------|------|------|------|------|------|------|------|------|
  | -168.7 | -181.1 | -88.9 | -0.4 | -16.3 | 103.9 | -31.8 | -12.4 | -28.9 | 135.0 |
  
  轨迹整体上升，但 cp6 出现峰值后回落，再回升至最终 135.0。

## 2. 骨架演进表
| iter | score | len | term (term/trunc) | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|------|-------|-----|-------------------|----------------------|----------------------|-----------------------|-----------------------|----------------------------|----------------------------|-------------------------|-------------------------|
| 1 | -32.35 | 1000 | 0/20, 20/20 | 0.00 | 0.0% | 3.76 | 100.0% | -3.96 | 100.0% | -0.49 | 100.0% |
| 2 | 107.13 | 775.6 | 11/20, 9/20 | 154.46 | 2.8% | 3.96 | 99.1% | -3.09 | 100.0% | -5.31 | 99.2% |

## 3. 数字标记
- **主导**：contact_bonus 的 magnitude_share = 92.5%（>80%），主导最终回报。
- **死组件**：无（所有组件 active_rate ≥ 2.8%）。
- **淹没**：
  - goal_proximity：active_rate = 99.1%，magnitude_share = 2.5%（<3%）。
  - orientation_penalty：active_rate = 100.0%，magnitude_share = 1.9%（<3%）。
- **全覆盖惩罚**（active_rate > 95% 且 signed_share 为负）：
  - orientation_penalty：active 100%，signed_share = -1.9%。
  - velocity_penalty：active 99.2%，signed_share = -3.2%。
- **量级对比**（每步均值绝对值，ep_sum ÷ len）：
  - 最大两个：contact_bonus（0.199）、velocity_penalty（0.00685）。
  - 最小两个：goal_proximity（0.0051）、orientation_penalty（0.00398）。
- **修改变化（iter 1 → iter 2）**：
  - contact_bonus：ep_sum 0.00 → 154.46（+154.46），active_rate 0.0% → 2.8%（+2.8pp）。
  - goal_proximity：ep_sum 3.76 → 3.96（+0.20），active_rate 100.0% → 99.1%（-0.9pp）。
  - orientation_penalty：ep_sum -3.96 → -3.09（+0.87），active_rate 不变 100.0%。
  - velocity_penalty：ep_sum -0.49 → -5.31（-4.82），active_rate 100.0% → 99.2%（-0.8pp）。

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum_mean） | 方向切换 |
|----------|-----------|------------------------|----------|
| 接近/进度信号 | goal_proximity（iter1 → iter2） | 3.76 → 3.96 | 无（始终保持delta改善量） |
| 着陆/接触信号 | contact_bonus（iter1 → iter2） | 0.00 → 154.46 | 无（始终为稀疏bonus） |
| 稳定性-姿态 | orientation_penalty（iter1 → iter2） | -3.96 → -3.09 | 无（始终为连续惩罚） |
| 稳定性-速度 | velocity_penalty（iter1 → iter2） | -0.49 → -5.31 | 门控方式：距离反比门控 → 指数高度门控（1次切换） |

（注：velocity_penalty 的权重同时从 0.05 增加到 0.5，属于同一修改内的参数变化，不计为额外切换。）