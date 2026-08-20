# 训练数据报告 — iter_01

## 1. 行为事实
- 得分：score = -214.16，range = [-254.45, -175.66]
- 终止方式：terminated = 20/20，truncated = 0/20  
  所有回合均以 early termination 条件结束（每回合均达到 early_terminal 判定，即 step<150 且 score<-50 的个数为 9/20，此处 terminated 统计可能代表环境定义的终止，但实际全部回合均在 182.5 步内自行结束）。
- 回合长度：len = 182.50（平均值）
- Checkpoint 轨迹：cp_40% score = 0.00，cp_80% score = 0.00，无变化。

## 2. 骨架演进表
| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active |
|------|-------|------|------|----------------------|----------------------|---------------------|---------------------|------------------------|------------------------|
| 1    | -214.16 | 182.50 | 20/0 | 0.152428 | 100.0% | 0.0 | 0.0% | 0.0 | 0.0% |

## 3. 数字标记
- **主导**：goal_approach 组件的 magnitude_share = 100.0%（>80%），为当前奖励结构唯一贡献者。
- **死组件**：soft_landing 与 crash_avoidance 的 active_rate = 0.0%（<2%），在评估 episode 中从未被触发。
- **淹没**：无（goal_approach 的 active_rate=100% 且 magnitude_share=100%，>90% 且 >3%，不满足淹没）。
- **全覆盖惩罚**：无组件满足 active_rate > 95% 且 signed_share 为负（crash_avoidance 未激活，无 negative signed_share 且 active_rate 为 0）。
- **量级对比**（每步均值）：
  - 最大：goal_approach = 0.152428 / 182.5 ≈ 0.0008356
  - 次大：soft_landing = 0.0
  - 最小：crash_avoidance = 0.0
  - 次小：无其他组件。
- **修改变化**：无上一轮数据，不做 Δ 列出。