# 训练数据报告（iter 03）

## 1. 行为事实
- 得分：score = -26.17，范围 [min=-55.96, max=13.34]
- 终止方式：terminated 0/20（成功 0），truncated 20/20（时间上限 20）
- 回合长度：len = 1000.00
- Checkpoint 轨迹：官方得分从 cp1=-135.0 → cp10=-24.4，先升后降，峰值 cp6=207.1

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|------|-------|-----|------|----------------------|----------------------|-----------------------|-----------------------|----------------------------|----------------------------|-------------------------|-------------------------|
| 1 | -32.35 | 1000.00 | T 0/20 Tru 20/20 | 0.000 | 0.0% | 3.758 | 100.0% | -3.959 | 100.0% | -0.486 | 100.0% |
| 2 | 107.13 | 775.60 | T 11/20 Tru 9/20 | 154.465 | 2.8% | 3.963 | 99.1% | -3.090 | 100.0% | -5.313 | 99.2% |
| 3 | -26.17 | 1000.00 | T 0/20 Tru 20/20 | 0.000 | 0.0% | 7.685 | 100.0% | -4.038 | 100.0% | -4.758 | 100.0% |

## 3. 数字标记
- **主导**：iter 2 contact_bonus magnitude_share = 92.5%（>80%），主导。
- **死组件**：
  - iter 1 contact_bonus active_rate = 0.0%（<2%）
  - iter 3 contact_bonus active_rate = 0.0%（<2%）
- **淹没**：iter 2 goal_proximity active_rate = 99.1% 且 magnitude_share = 2.5%（<3%）；orientation_penalty active_rate = 100.0% 且 magnitude_share = 1.9%（<3%）
- **全覆盖惩罚**：
  - iter 2 velocity_penalty active_rate = 99.2% 且 signed_share = -3.2%；orientation_penalty active_rate = 100.0% 且 signed_share = -1.9%
  - iter 3 velocity_penalty active_rate = 100.0% 且 signed_share = -27.9%；orientation_penalty active_rate = 100.0% 且 signed_share = -23.7%
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - iter 1 最大：orientation_penalty（0.00396），goal_proximity（0.00376）；最小：velocity_penalty（0.00049），contact_bonus（0.00000）
  - iter 2 最大：contact_bonus（0.19915），velocity_penalty（0.00685）；最小：goal_proximity（0.00511），orientation_penalty（0.00399）
  - iter 3 最大：goal_proximity（0.00768），velocity_penalty（0.00476）；最小：orientation_penalty（0.00404），contact_bonus（0.00000）
- **修改变化**（当前轮 vs 上一轮）：
  - contact_bonus: ep_sum 154.47 → 0.00（Δ -154.47），active_rate 2.8% → 0.0%（Δ -2.8pp）
  - goal_proximity: ep_sum 3.96 → 7.69（Δ +3.72），active_rate 99.1% → 100.0%（Δ +0.9pp）
  - velocity_penalty: ep_sum -5.31 → -4.76（Δ +0.55，惩罚减轻），active_rate 99.2% → 100.0%（Δ +0.8pp）
  - orientation_penalty: ep_sum -3.09 → -4.04（Δ -0.95，惩罚加重），active_rate 100.0% → 100.0%（Δ 0pp）

## 4. 语义演化（跨轮组件功能追踪）
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | v1_contact_bonus → v2_contact_bonus → v3_contact_bonus | 0.000 → 154.465 → 0.000 | 无切换（始终稀疏bonus） |
| 接近/进度信号 | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity | 3.758 → 3.963 → 7.685 | 无切换（始终改善量/连续奖励） |
| 稳定性/约束信号 | v1_velocity_penalty, v1_orientation_penalty → v2_velocity_penalty, v2_orientation_penalty → v3_velocity_penalty, v3_orientation_penalty | v: -0.486, -5.313, -4.758; o: -3.959, -3.090, -4.038 | velocity_penalty：门控类型切换1次（曼哈顿距离门控→指数高度门控） |