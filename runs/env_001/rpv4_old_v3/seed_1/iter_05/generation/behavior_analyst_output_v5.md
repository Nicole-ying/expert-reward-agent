# 行为事实
- 得分：score = -111.06，区间 [−130.75, −95.06]
- 终止方式：terminated 20/20（所有episode均因达到终止条件结束），truncated 0/20（无episode因步数上限截断）
- 回合长度：平均 len = 68.40 步
- Checkpoint轨迹（部分数据）：训练进度 40% 时官方得分 0.00，80% 时官方得分 0.00，保持持平

# 骨架演进表

| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active | contact_reward_ep_sum | contact_reward_active |
|------|--------|-------|------|------------------------|----------------------|----------------------|----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 1 | -214.16 | 182.50 | 20/0 | 0.1524 | 1.00 | 0.0000 | 0.00 | 0.00 | 0.00 | — | — | — | — |
| 2 | -109.58 | 68.45 | 20/0 | 1.1202 | 1.00 | -19.0910 | 1.00 | 0.00 | 0.00 | — | — | — | — |
| 3 | -115.21 | 68.35 | 20/0 | 1.1188 | 1.00 | -20.7435 | 1.00 | — | — | 25.00 | 0.004 | — | — |
| 4 | -111.06 | 68.40 | 20/0 | 1.1204 | 1.00 | -19.2969 | 1.00 | — | — | — | — | 1.15 | 0.017 |

*注：term列格式为“terminated数/truncated数”；“—”表示该组件在本轮奖励函数中不存在。

# 数字标记（基于本轮 iter=4）

- **主导**：soft_landing magnitude_share = 89.3% > 80%
- **全覆盖惩罚**：soft_landing active_rate = 100% > 95% 且 signed_share = −89.3% 为负
- **死组件**：contact_reward active_rate = 1.7% < 2%
- **量级对比（每步均值）**  
  - 最大：contact_reward = 1.15/68.4 ≈ 0.01681，goal_approach = 1.1204/68.4 ≈ 0.01638  
  - 最小：soft_landing = −19.2969/68.4 ≈ −0.2821，goal_approach ≈ 0.01638  
- **修改变化（相对于 iter=3）**  
  - soft_landing：ep_sum Δ = +1.4466，active Δ = 0.00  
  - goal_approach：ep_sum Δ = +0.0016，active Δ = 0.00  
  - landing_success 已移除，contact_reward 为新增组件，无Δ计算