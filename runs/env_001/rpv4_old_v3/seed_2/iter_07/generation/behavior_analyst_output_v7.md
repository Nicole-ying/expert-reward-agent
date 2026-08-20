# 训练数据报告

## 1. 行为事实
- 得分：score = -111.18，range = [-124.08, -95.17]
- 终止方式：terminated = 20/20（全部接触地面导致结束），truncated = 0/20
- 回合长度：len = 68.45 步
- Checkpoint 轨迹：官方得分全程为 0.00（40% 步 = 0.00，80% 步 = 0.00）

## 2. 骨架演进表
| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | crash_penalty_ep_sum | crash_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -155.98 | 154.40 | 20/20 | 1.582 | 100.0% | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | 0.000 | 0.0% | 0.000 | 0.0% |
| 2 | -119.51 | 68.30 | 20/20 | 5.583 | 100.0% | -0.071 | 5.2% | -0.205 | 100.0% | 0.000 | 0.0% | 5.000 | 0.1% | 0.000 | 0.0% |
| 3 | -203.45 | 154.15 | 20/20 | 1.488 | 100.0% | -1.472 | 47.7% | -2.460 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | 0.000 | 0.0% |
| 4 | -114.38 | 68.45 | 20/20 | 5.597 | 100.0% | -0.097 | 7.1% | -0.087 | 100.0% | 0.000 | 0.0% | 30.000 | 0.4% | 0.000 | 0.0% |
| 5 | -46.51 | 740.15 | 7/20 | 5.105 | 99.9% | -13.469 | 91.0% | -6.828 | 100.0% | 0.000 | 0.0% | 1195.000 | 1.6% | 0.000 | 0.0% |
| 6 | -111.18 | 68.45 | 20/20 | 5.603 | 100.0% | -0.067 | 4.9% | -0.020 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | 0.000 | 0.0% |

## 3. 数字标记
- **主导**：`progress_shaping` magnitude_share = 98.5%（> 80%）
- **死组件**：`success_landing_bonus` active_rate = 0.0%（< 2%）
- **淹没**：`angle_penalty` active_rate = 100.0%（> 90%），magnitude_share = 0.3%（< 3%）
- **全覆盖惩罚**：`angle_penalty` active_rate = 100.0%（> 95%），signed_share = -0.3%（负值）
- **量级对比**（每步均值 = ep_sum ÷ len，len=68.45）：
  - 最大两个：`progress_shaping` = 0.0819 / step，`action_cost` = -0.00098 / step
  - 最小两个：`angle_penalty` = -0.000293 / step，`success_landing_bonus` = 0.0 / step
- **修改变化**（iter 5 → iter 6）：
  - `progress_shaping` ep_sum Δ = +0.498，active_rate Δ = +0.1pp
  - `action_cost` ep_sum Δ = +13.402，active_rate Δ = -86.1pp
  - `angle_penalty` ep_sum Δ = +6.808，active_rate Δ = 0.0pp
  - `success_landing_bonus` ep_sum Δ = -1195.0，active_rate Δ = -1.6pp

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1–v6: `progress_shaping` | -155.98 → -119.51 → -203.45 → -114.38 → -46.51 → -111.18 | 无切换（始终为连续 shaping） |
| 着陆/接触信号 | v1: `landing_velocity_penalty`；v2: `success_landing_bonus`；v3: `success_landing_bonus` + `crash_penalty`；v4–v6: `success_landing_bonus` | -155.98 → -119.51 → -203.45 → -114.38 → -46.51 → -111.18 | 连续惩罚→稀疏bonus：1次切换（v1→v2） |
| 稳定性/约束 | v1–v6: `angle_penalty` + `action_cost` | 同上 | 无切换（角惩罚始终二次型，动作代价始终常数） |