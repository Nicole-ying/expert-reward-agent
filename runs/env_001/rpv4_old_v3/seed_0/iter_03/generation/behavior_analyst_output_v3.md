# 训练数据报告 — 基于 iter_02（本轮）与 iter_01（历史）

## 1. 行为事实
- **得分**  
  - iter_02：score=217.21，range=[61.62, 265.55]  
  - iter_01：score=-40.04，range=[-133.28, 15.32]  
- **终止方式**  
  - iter_02：terminated=18/20（成功/失败），truncated=2/20（时间上限）  
  - iter_01：terminated=20/20，truncated=0/20  
- **回合长度**  
  - iter_02：len=484.10  
  - iter_01：len=154.05  
- **Checkpoint 轨迹**（iter_02 官方环境得分）  
  - pct 40%：0.00  
  - pct 80%：0.00  
  - 变化趋势：持平（全程 0.00）

## 2. 骨架演进表
| iter | score | len | term | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active | goal_proximity_ep_sum | goal_proximity_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active | landing_success_reward_ep_sum | landing_success_reward_active |
|------|-------|-----|------|----------------------------|---------------------------|----------------------------|---------------------------|------------------------|-----------------------|------------------------|-----------------------|-----------------------------|-----------------------------|-------------------------------|-------------------------------|
| 1    | -40.04 | 154.05 | terminated=20/20, truncated=0/20 | 0.000 | 0.0% | 0.000 | 0.0% | 2.139 | 100.0% | -0.308 | 100.0% | -0.053 | 1.5% | — | — |
| 2    | 217.21 | 484.10 | terminated=18/20, truncated=2/20 | 0.000 | 0.0% | 0.000 | 0.0% | 2.568 | 97.6% | -0.968 | 100.0% | — | — | 36.450 | 7.5% |

> 注：iter_01 的 `soft_landing_penalty` 在 iter_02 中被替换为 `landing_success_reward`，故二者在同一行不可共存，分别列示。

## 3. 数字标记
- **主导**  
  - iter_02：`landing_success_reward` magnitude_share = 90.6% > 80%，主导。
- **死组件**  
  - iter_01 与 iter_02：`crash_prevention_x`、`crash_prevention_y` active_rate 均为 0.0% < 2%。  
  - iter_01：`soft_landing_penalty` active_rate = 1.5% < 2%。
- **淹没**  
  - 无。（iter_02：`goal_proximity` active_rate = 97.6% > 90%，但 magnitude_share = 7.0% > 3%，不满足淹没条件。）
- **全覆盖惩罚**  
  - iter_02：`living_penalty` active_rate = 100.0% > 95% 且 signed_share = -2.4% < 0，全覆盖惩罚。
- **量级对比**（iter_02 每步均值 = episode_sum_mean ÷ len）  
  - 最大两个  
    - `landing_success_reward`：36.450 ÷ 484.10 ≈ 0.0753  
    - `goal_proximity`：2.568 ÷ 484.10 ≈ 0.0053  
  - 最小两个  
    - `crash_prevention_x`：0.000  
    - `crash_prevention_y`：0.000  
    - （`living_penalty`：-0.968 ÷ 484.10 ≈ -0.002）
- **修改变化**（iter_01 → iter_02 的 Δ）
  - `goal_proximity`：ep_sum +0.429，active_rate -2.4pp  
  - `living_penalty`：ep_sum -0.660，active_rate 0.0pp  
  - `crash_prevention_x`：ep_sum 0.000，active_rate 0.0pp  
  - `crash_prevention_y`：ep_sum 0.000，active_rate 0.0pp  
  - `landing_success_reward`（新增）：ep_sum +36.450，active_rate +7.5pp  
  - `soft_landing_penalty`（移除）：ep_sum +0.053（由 -0.053 变为不适用），active_rate -1.5pp