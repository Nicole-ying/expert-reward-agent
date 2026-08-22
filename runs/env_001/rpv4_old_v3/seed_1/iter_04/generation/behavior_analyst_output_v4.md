# 训练数据报告

## 1. 行为事实
- **得分**：score = -115.206637，范围 [-142.247427, -100.640778]
- **终止方式**：terminated = 20/20（全部因 early_terminal 终止），truncated = 0/20
- **回合长度**：len = 68.35
- **Checkpoint 轨迹**：官方得分从 cp1 (40%) = 0.00 到 cp2 (80%) = 0.00（持平）
- **early_terminal(<150步且分数<-50)**：20/20

## 2. 骨架演进表

| iter | score       | len    | term    | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active |
|------|-------------|--------|---------|----------------------|----------------------|---------------------|---------------------|------------------------|------------------------|------------------------|------------------------|
| 1    | -214.161811 | 182.50 | 20/20   | 0.152428             | 100%                 | 0.000000            | 0%                  | 0.000000               | 0%                     | —                      | —                      |
| 2    | -109.576065 | 68.45  | 20/20   | 1.120194             | 100%                 | -19.091045          | 100%                | 0.000000               | 0%                     | —                      | —                      |
| 3    | -115.206637 | 68.35  | 20/20   | 1.118841             | 100%                 | -20.743517          | 100%                | —                      | —                      | 25.000000              | 0.4%                   |

（注：“—”表示对应组件在该轮奖励函数中不存在，ep_sum 和 active_rate 视为 0。）

## 3. 数字标记

- **主导**：无（最大 magnitude_share = 53.3%，未超过 80%）
- **死组件**：`landing_success`，active_rate = 0.4% < 2%
- **淹没**：`goal_approach`，active_rate = 100% > 90%，但 magnitude_share = 2.5% < 3%
- **全覆盖惩罚**：`soft_landing`，active_rate = 100% > 95%，且 signed_share = -44.2%（负值）
- **量级对比**（每步均值 = ep_sum ÷ len，len=68.35）：
  - 最大两个：`landing_success` (+0.3658)，`goal_approach` (+0.0164)
  - 最小两个：`soft_landing` (-0.3035)，`goal_approach` (+0.0164)
- **修改变化**（iter_03 相对于 iter_02）：
  - `goal_approach`：ep_sum Δ = -0.00135，active_rate Δ = 0%
  - `soft_landing`：ep_sum Δ = -1.6525，active_rate Δ = 0%
  - `crash_avoidance`：ep_sum Δ = 0（eliminated），active_rate Δ = 0%
  - `landing_success`：ep_sum Δ = +25.0（新增），active_rate Δ = +0.4%