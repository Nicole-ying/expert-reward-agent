# 训练数据报告

## 1. 行为事实
- 得分：score = 212.94，range = [69.66, 274.05]
- 终止方式：terminated 16/20（成功着陆），truncated 4/20（超时）
- 回合长度：len = 511.90
- Checkpoint 轨迹（官方得分）：
  - 40% 步数：score_mean = 0.00
  - 80% 步数：score_mean = 0.00
  → 从 40% 到 80% 官方得分均为 0，持平

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | landing_success_reward_ep_sum | landing_success_reward_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active |
|------|--------|--------|---------------------------------|------------------------|------------------------|-------------------------------|-------------------------------|------------------------|------------------------|-----------------------------|-----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| 1    | -40.04 | 154.05 | terminated 20/20, truncated 0   | 2.1388                 | 100.0%                 | –                             | –                             | -0.3081                | 100.0%                 | -0.0526                     | 1.5%                        | 0.0                        | 0.0%                       | 0.0                        | 0.0%                       |
| 2    | 217.21 | 484.10 | terminated 18/20, truncated 2   | 2.5678                 | 97.6%                  | 36.45                         | 7.5%                          | -0.9682                | 100.0%                 | –                           | –                           | 0.0                        | 0.0%                       | 0.0                        | 0.0%                       |
| 3    | 212.94 | 511.90 | terminated 16/20, truncated 4   | 2.6382                 | 97.6%                  | 37.25                         | 7.3%                          | -1.0238                | 100.0%                 | –                           | –                           | –                          | –                          | –                          | –                          |

*注：iter2 的 crash_prevention_x / y 仍保留在代码中，但每回合均值为 0，active_rate 为 0；iter3 代码中已移除这两个组件，标记为“–”。*

## 3. 数字标记

- **主导**：landing_success_reward 的 magnitude_share = 90.7% > 80%
- **死组件**：本轮（iter3）无活跃组件 active_rate < 2%
- **淹没**：living_penalty 的 active_rate = 100.0% > 90%，magnitude_share = 2.5% < 3%
- **全覆盖惩罚**：living_penalty 的 active_rate = 100.0% > 95% 且 signed_share = -2.5%（负）
- **量级对比**（ep_sum ÷ len，按本组数值）：
  - 最大的两个：
    1. landing_success_reward: 37.25 / 511.9 ≈ 0.0728
    2. goal_proximity: 2.6382 / 511.9 ≈ 0.00515
  - 最小的两个：
    1. living_penalty: -1.0238 / 511.9 ≈ -0.00200
    2. goal_proximity: ≈ 0.00515（与上面并列第二；因仅有三个组件，故列出该值）
- **修改变化**（iter3 vs iter2，公共组件）：
  - landing_success_reward: ep_sum Δ = +0.80，active_rate Δ = -0.2 个百分点
  - goal_proximity: ep_sum Δ = +0.0704，active_rate Δ = 0.0 个百分点
  - living_penalty: ep_sum Δ = -0.0556，active_rate Δ = 0.0 个百分点