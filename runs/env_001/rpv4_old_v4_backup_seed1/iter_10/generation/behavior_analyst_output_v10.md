# 数据报告

## 1. 行为事实
- 得分：score=3.04，range=[-51.59, 201.81]
- 终止方式：terminated 2/20（成功），truncated 18/20（时间上限）
- 回合长度：len=965.95
- Checkpoint 轨迹：官方得分从 cp1=-209.1 → cp2=-153.6 → cp3=-88.0 → cp4=-32.7 → cp5=-56.6 → cp6=-13.2 → cp7=-42.1 → cp8=-18.2 → cp9=-23.5 → cp10=13.9（整体上升，有波动）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | soft_landing_envelope_ep_sum | soft_landing_envelope_active | contact_bonus_ep_sum | contact_bonus_active | velocity_penalty_ep_sum | velocity_penalty_active | upright_penalty_ep_sum | upright_penalty_active | landing_approach_ep_sum | landing_approach_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | -76.56 | 975.15 | 1/20 | 0.64 | 100.0% | — | — | 0.00 | 0.0% | -12.56 | 100.0% | -3.71 | 100.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1.84 | 98.8% | — | — | 1075.63 | 5.2% | -9.52 | 99.0% | -7.30 | 100.0% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1.77 | 99.9% | — | — | 1011.47 | 8.2% | -9.10 | 99.9% | -11.45 | 100.0% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 2.92 | 100.0% | — | — | 0.00 | 0.0% | -7.84 | 100.0% | -5.11 | 100.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 4.29 | 100.0% | — | — | 0.00 | 0.0% | -10.12 | 100.0% | -3.60 | 100.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | 1.77 | 100.0% | — | — | — | — | -2.39 | 100.0% | -3.08 | 100.0% | 0.97 | 98.3% |
| 7 | -5.37 | 964.10 | 2/20 | 4.75 | 100.0% | — | — | 95.90 | 0.7% | -4.62 | 100.0% | -5.45 | 100.0% | — | — |
| 8 | -76.20 | 1000.00 | 0/20 | 2.73 | 100.0% | — | — | 0.00 | 0.0% | -5.31 | 100.0% | -4.19 | 100.0% | — | — |
| 9 | 3.04 | 965.95 | 2/20 | 5.03 | 100.0% | 1670.93 | 100.0% | — | — | -9.01 | 100.0% | -5.31 | 100.0% | — | — |

## 3. 数字标记

- **主导**：soft_landing_envelope magnitude_share=98.8%（>80%）

- **死组件**：无。所有组件 active_rate 均为 100.0%

- **淹没**：goal_proximity active_rate=100% 但 magnitude_share=0.4%（<3%）；upright_penalty active_rate=100% 但 magnitude_share=0.3%（<3%）；velocity_penalty active_rate=100% 但 magnitude_share=0.5%（<3%）

- **全覆盖惩罚**：无。无组件 active_rate > 95% 且 signed_share 为负

- **量级对比**：
  - 每步均值（ep_sum ÷ len）最大：soft_landing_envelope=1.730
  - 每步均值（ep_sum ÷ len）第二大：goal_proximity=0.005
  - 每步均值（ep_sum ÷ len）最小：velocity_penalty=-0.009
  - 每步均值（ep_sum ÷ len）第二小：upright_penalty=-0.006

- **修改变化（iter 9 vs iter 8）**：
  | component | Δ ep_sum | Δ active_rate |
  ||---|---:|---:|
  | goal_proximity | +2.30 | 0pp |
  | velocity_penalty | -3.70 | 0pp |
  | upright_penalty | -1.12 | 0pp |
  | contact_bonus（iter 8）→ soft_landing_envelope（iter 9） | +1670.93 | +100.0pp |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | v1_contact_bonus(死) → v2_contact_bonus → v3_contact_bonus → v4_contact_bonus(死) → v5_contact_bonus(死) → v6_landing_approach → v7_contact_bonus → v8_contact_bonus(死) → v9_soft_landing_envelope | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 稀疏bonus↔连续奖励：3次切换（v6, v7, v9） |
| 接近/进度信号 | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity(淹没) → v5_goal_proximity(淹没) → v6_goal_proximity → v7_goal_proximity → v8_goal_proximity(淹没) → v9_goal_proximity(淹没) | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 线性delta→逆距离delta→线性delta：2次切换（v4, v5）；淹没状态：v4/v5/v8/v9 |
| 稳定性/约束信号（速度） | v1_velocity_penalty → v2_velocity_penalty → v3_velocity_penalty → v4_velocity_penalty → v5_velocity_penalty → v6_velocity_penalty → v7_velocity_penalty → v8_velocity_penalty → v9_velocity_penalty | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 门控宽度切换：k=∞→k=2.0→k=1.0→k=1.5（3次切换） |
| 稳定性/约束信号（姿态） | v1_upright_penalty → v2_upright_penalty → v3_upright_penalty → v4_upright_penalty → v5_upright_penalty → v6_upright_penalty → v7_upright_penalty → v8_upright_penalty → v9_upright_penalty | -77 → 64 → 21 → -10 → -33 → -82 → -5 → -76 → 3 | 无切换（始终为二次型姿态角+角速度惩罚） |