# 训练数据报告 — iter_04 (v4)

## 1. 行为事实

- **得分**：score=183.52，range=[90.97, 220.09]
- **终止方式**：terminated=19/20（成功着陆），truncated=1/20（时间上限）
- **回合长度**：len=710.25
- **Checkpoint 轨迹**：官方得分从 cp1=-102.4 → cp3=-37.0 → cp9=2.3 → cp10=187.5（持续上升，cp9→cp10 跳跃 +185.2）
- **奖励错误**：0

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | descent_incentive_ep_sum | descent_incentive_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -32.35 | 1000.00 | 0/0/20 | 0.000 | 0.0% | 3.758 | 100.0% | -0.486 | 100.0% | -3.959 | 100.0% | — | — |
| 2 | 107.13 | 775.60 | 11/9/0 | 154.465 | 2.8% | 3.963 | 99.1% | -5.313 | 99.2% | -3.090 | 100.0% | — | — |
| 3 | -26.17 | 1000.00 | 0/20/0 | 0.000 | 0.0% | 7.685 | 100.0% | -4.758 | 100.0% | -4.038 | 100.0% | — | — |
| 4 | 183.52 | 710.25 | 19/0/1 | 114.083 | 4.9% | 7.951 | 98.1% | -5.736 | 98.4% | -3.386 | 100.0% | 0.779 | 72.4% |

## 3. 数字标记

### 主导
- **contact_bonus**：magnitude_share=86.3%（>80%），主导总奖励量级

### 死组件
- 无（所有组件 active_rate ≥ 4.9%）

### 淹没
- 无（无组件同时满足 active_rate>90% 且 magnitude_share<3%）

### 全覆盖惩罚
- **orientation_penalty**：active_rate=100%（>95%）且 signed_share 为负（-2.6%）

### 量级对比（每步均值 = ep_sum ÷ len，从 iter_04 计算）
- **最大两个**：contact_bonus=0.1606，goal_proximity=0.0112
- **最小两个**：descent_incentive=0.0011，orientation_penalty=-0.0048

### 修改变化（iter_03 → iter_04 的 Δ）
| component | Δ ep_sum | Δ active_rate |
|---|---|---|
| contact_bonus | 0.000 → 114.083 (+114.083) | 0.0% → 4.9% (+4.9pp) |
| goal_proximity | 7.685 → 7.951 (+0.266) | 100.0% → 98.1% (-1.9pp) |
| velocity_penalty | -4.758 → -5.736 (-0.978) | 100.0% → 98.4% (-1.6pp) |
| orientation_penalty | -4.038 → -3.386 (+0.652) | 100.0% → 100.0% (0pp) |
| descent_incentive | 新增 | 新增 |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| **着陆/接触信号** | v1_contact_bonus(死) → v2_contact_bonus → v3_contact_bonus(死) → v4_contact_bonus | -32.35 → 107.13 → -26.17 → 183.52 | 稀疏bonus：v1失效→v2活跃→v3失效→v4活跃，**2次切换** |
| **接近/进度信号** | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity + v4_descent_incentive | 同上 | 改善量(Δ)：v4新增descent_incentive为连续状态值形式，**1次新增** |
| **稳定性/约束信号-速度** | v1_velocity_penalty → v2_velocity_penalty → v3_velocity_penalty → v4_velocity_penalty | 同上 | 门控形式：v1曼哈顿距离门控→v2指数高度门控（沿用至v4），**1次切换** |
| **稳定性/约束信号-姿态** | v1_orientation_penalty → v2_orientation_penalty → v3_orientation_penalty → v4_orientation_penalty | 同上 | 无切换（四轮形式一致：|angle| + angvel²） |