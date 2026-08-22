# 训练数据报告

## 1. 行为事实
- **得分**：score = -56.068，range = [-134.812, 71.239]  
- **终止方式**：terminated 20/20（均为成功/失败终止），truncated 0/20（无时间截断）  
- **回合长度**：len = 109.20  
- **Checkpoint 轨迹**（每10%步数的官方得分）：  
  - 40%：0.00  
  - 80%：0.00  
  （未提供其他百分比的数据）  
- **早期终止**（步数<150且 score<-50）：12/20 集

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | landing_success_reward_ep_sum | landing_success_reward_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active |
|------|--------|----------|------|------------------------|-----------------------|-------------------------------|-------------------------------|------------------------|-----------------------|-----------------------------|-----------------------------|----------------------------|---------------------------|----------------------------|---------------------------|
| 1    | -40.04 | 154.05   | 20/0 | 2.139                  | 100.0%                | -                             | -                             | -0.308                 | 100.0%                | -0.053                      | 1.5%                        | 0.0                        | 0.0%                      | 0.0                        | 0.0%                      |
| 2    | 217.21 | 484.10   | 18/2 | 2.568                  | 97.6%                 | 36.450                        | 7.5%                          | -0.968                 | 100.0%                | -                           | -                           | 0.0                        | 0.0%                      | 0.0                        | 0.0%                      |
| 3    | 212.94 | 511.90   | 16/4 | 2.638                  | 97.6%                 | 37.250                        | 7.3%                          | -1.024                 | 100.0%                | -                           | -                           | -                          | -                         | -                          | -                         |
| 4    | -56.07 | 109.20   | 20/0 | 2.592                  | 100.0%                | 0.650                         | 0.6%                          | -0.546                 | 100.0%                | -                           | -                           | -                          | -                         | -                          | -                         |

## 3. 数字标记
- **主导**：无组件 magnitude_share > 80%  
- **死组件**：`landing_success_reward`（active_rate = 0.6% < 2%）  
- **淹没**：无（`goal_proximity`、`living_penalty` 虽然 active_rate > 90%，但 magnitude_share 分别为 69.3% 和 14.0%，均 > 3%）  
- **全覆盖惩罚**：`living_penalty`（active_rate = 100% > 95% 且 signed_share = -14.0%，为负）  
- **量级对比**（每步均值 = ep_sum / len）：  
  - 从大到小排列：`goal_proximity`（+0.0237）、`landing_success_reward`（+0.0060）、`living_penalty`（-0.0050）  
  - 最大的两个组件：`goal_proximity`、`landing_success_reward`  
  - 最小的两个组件：`landing_success_reward`、`living_penalty`（注：`landing_success_reward` 同时出现在两组中）  
- **修改变化**（相对于 iter 3）：  
  - `goal_proximity`：ep_sum Δ = -0.046，active_rate Δ = +2.4 个百分点  
  - `landing_success_reward`：ep_sum Δ = -36.600，active_rate Δ = -6.7 个百分点  
  - `living_penalty`：ep_sum Δ = +0.478（负值上升，即惩罚减弱），active_rate Δ = 0.0 个百分点  

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 (ep_sum) | 方向切换 |
|----------|-----------|------------------|----------|
| 接近/进度信号 | goal_proximity → goal_proximity → goal_proximity → goal_proximity | 2.14 → 2.57 → 2.64 → 2.59 | 无 |
| 着陆/接触信号 | （无）→ landing_success_reward → landing_success_reward → landing_success_reward | - → 36.45 → 37.25 → 0.65 | 无（始终为稀疏 bonus） |
| 稳定性/约束信号 | soft_landing_penalty + crash_prevention_x + crash_prevention_y → crash_prevention_x + crash_prevention_y → （无）→ （无） | v1: -0.05 (soft) + 0.0 (x) + 0.0 (y)；v2: 0.0 + 0.0；v3: -；v4: - | 无 |
| 时间/效率惩罚 | living_penalty → living_penalty → living_penalty → living_penalty | -0.31 → -0.97 → -1.02 → -0.55 | 无 |