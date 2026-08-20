# 训练数据报告（iter=09）

## 1. 行为事实

- **得分**：score=145.168, range=[33.532, 288.929]
- **终止方式**：terminated=9/20（腿部触地终止），truncated=11/20（时间上限）
- **回合长度**：len=718.800
- **Checkpoint 轨迹**：官方得分 cp4=0.00 → cp8=0.00（持平）

## 2. 骨架演进表

| iter | score | len | term | action_cost/fuel_penalty_ep_sum | active | angle_penalty/angle_stab_ep_sum | active | landing_vel_penalty_ep_sum | active | progress_shaping/progress_ep_sum | active | crash_penalty_ep_sum | active | success_landing_bonus_ep_sum | active | vel_damping_ep_sum | active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -155.98 | 154.40 | 20/0 | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | 1.582 | 100.0% | — | — | — | — | — | — |
| 2 | -119.51 | 68.30 | 20/0 | -0.071 | 5.2% | -0.205 | 100.0% | — | — | 5.583 | 100.0% | — | — | 5.000 | 0.1% | — | — |
| 3 | -203.45 | 154.15 | 20/0 | -1.472 | 47.7% | -2.460 | 100.0% | — | — | 1.488 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | — | — |
| 4 | -114.38 | 68.45 | 20/0 | -0.097 | 7.1% | -0.087 | 100.0% | — | — | 5.597 | 100.0% | — | — | 30.000 | 0.4% | — | — |
| 5 | -46.51 | 740.15 | 7/13 | -13.469 | 91.0% | -6.828 | 100.0% | — | — | 5.105 | 99.9% | — | — | 1195.000 | 1.6% | — | — |
| 6 | -111.18 | 68.45 | 20/0 | -0.067 | 4.9% | -0.020 | 100.0% | — | — | 5.603 | 100.0% | — | — | 0.000 | 0.0% | — | — |
| 7 | -98.93 | 69.60 | 20/0 | -0.178 | 12.8% | -0.221 | 100.0% | — | — | 5.553 | 100.0% | — | — | 5.000 | 0.1% | — | — |
| 8 | 26.45 | 206.50 | 20/0 | -1.232 | 59.6% | -0.263 | 100.0% | — | — | 2.667 | 100.0% | — | — | — | — | -1.577 | 99.9% |
| 9 | 145.17 | 718.80 | 9/11 | -4.268 | 59.4% | -0.748 | 100.0% | — | — | 2.638 | 99.2% | — | — | 5330.000 | 7.4% | -2.022 | 99.2% |

注：iter=8-9 使用新命名体系（fuel_penalty 对应旧 action_cost，angle_stab 对应旧 angle_penalty，progress 对应旧 progress_shaping，新增 vel_damping）。

## 3. 数字标记

- **主导**：success_landing_bonus magnitude_share=99.8% > 80%
- **死组件**：无（所有组件 active_rate ≥ 7.4%）
- **淹没**：无（无组件同时满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：无（无组件同时满足 active_rate > 95% 且 signed_share < 0）
- **量级对比**（每步均值 = ep_sum ÷ len，len=718.80）：
  - 最大：success_landing_bonus (7.415)，progress (0.00367)
  - 最小：fuel_penalty (-0.00594)，angle_stab (-0.00104)
- **修改变化**（iter=08 → iter=09）：
  | component | Δ ep_sum | Δ active_rate | Δ 每步均值 |
  |---|---|---|---|
  | progress | -0.029 | -0.8pp | -0.00128 |
  | vel_damping | -0.445 | -0.7pp | -0.00973 |
  | angle_stab | -0.485 | 0.0pp | -0.00149 |
  | fuel_penalty | -3.036 | -0.2pp | -0.00597 |
  | success_landing_bonus | +5330.0（新增） | +7.4pp | +7.415（新增） |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| 接近/进度信号 | v1_progress_shaping → v2-7_progress_shaping → v8-9_progress | -156→-119→-203→-114→-47→-111→-99→26→145 | 无切换（始终连续改善量，始终全高度，始终势能shaping） |
| 着陆/接触信号 | v1_landing_velocity_penalty(死) → v2_success_landing_bonus(死) → v3_success_landing_bonus(死) → v4_success_landing_bonus(死) → v5_success_landing_bonus(死) → v6_success_landing_bonus(死) → v7_success_landing_bonus(死) → v9_success_landing_bonus | -156→-119→-203→-114→-47→-111→-99→145 | 连续奖励→稀疏bonus：1次（v1→v2）；稀疏bonus↔连续奖励：0次；any_contact↔both_legs：1次（v7→v9）；状态值↔事件触发：1次（v5→v6）；阈值波动≥3次（0.1→0.2→0.4→0.3→0.4→0.4） |
| 稳定性/约束-姿态 | v1_angle_penalty → v2-7_angle_penalty → v8-9_angle_stab | -156→-119→-203→-114→-47→-111→-99→26→145 | 仅角度→角度+角速度：1次（v7→v8） |
| 稳定性/约束-速度 | v8-9_vel_damping | 26→145 | 无历史（v8新增） |
| 效率/约束-能耗 | v1-7_action_cost → v8-9_fuel_penalty | -156→-119→-203→-114→-47→-111→-99→26→145 | 权重不变（均为0.01-0.02量级） |
| 失败惩罚 | v3_crash_penalty(死) | -203 | 仅在v3存在，立即被移除，无切换 |