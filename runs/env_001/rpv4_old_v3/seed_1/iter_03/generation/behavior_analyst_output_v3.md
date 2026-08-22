## 1. 行为事实
- 得分：score = -109.576，范围 [-123.894, -95.059]
- 终止方式：全部 early terminal（score < -50 且 steps < 150），terminated = 20/20，truncated = 0/20；所有 episode 在达到 max steps 前因成功/失败而终止
- 回合长度：平均 68.45 步
- Checkpoint 轨迹：仅提供 40% 和 80% 步数处的得分，均为 0.00（无其他步数数据）

## 2. 骨架演进表
| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active |
|------|--------|------|-------|-----------------------|-----------------------|---------------------|---------------------|------------------------|-------------------------|
| 1    | -214.16 | 182.50 | 20/0 | 0.152 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% |
| 2    | -109.58 | 68.45  | 20/0 | 1.120 | 100.0% | -19.091 | 100.0% | 0.000 | 0.0% |

## 3. 数字标记
- **主导**：soft_landing 的 magnitude_share = 94.3% (> 80%)，占据奖励量级的绝对主导。
- **死组件**：crash_avoidance 的 active_rate = 0.0% (< 2%)，完全未激活。
- **淹没**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。
- **全覆盖惩罚**：soft_landing 的 active_rate = 100.0% (> 95%)，且 signed_share = -94.3%（负）。
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - goal_approach: 1.120194 / 68.45 ≈ 0.0164
  - soft_landing: -19.091045 / 68.45 ≈ -0.2789
  - crash_avoidance: 0.0
  - 最大的两个：goal_approach (0.0164)，crash_avoidance (0.0000)
  - 最小的两个：soft_landing (-0.2789)，crash_avoidance (0.0000)
- **修改变化**（与上一轮 iter_01 对比的 Δ）：
  - goal_approach ep_sum: +0.9678，active_rate: 0.0%
  - soft_landing ep_sum: -19.0910，active_rate: +100.0%
  - crash_avoidance ep_sum: 0.0，active_rate: 0.0%