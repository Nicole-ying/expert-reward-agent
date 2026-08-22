## 1. 行为事实
- 得分：score=225.88，range=[199.06, 261.87]
- 终止方式：terminated=20/20（全部正常结束），truncated=0/20（无因步数上限截断）
- 回合长度：len=479.70
- Checkpoint 轨迹：无数据（本迭代未提供 Checkpoint 评估）

## 2. 骨架演进表
| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | contact_reward_ep_sum | contact_reward_active | landing_speed_penalty_ep_sum | landing_speed_penalty_active | settlement_penalty_ep_sum | settlement_penalty_active |
|------|-------|-----|------|------------------------|------------------------|-----------------------|-----------------------|-----------------------------|------------------------------|---------------------------|---------------------------|
| 1    | 225.88 | 479.70 | terminated=20/20, truncated=0/20 | 346.76 | 100.0% | 70.21 | 7.8% | -56.49 | 99.8% | -0.33 | 7.8% |

## 3. 数字标记
- **全覆盖惩罚**：landing_speed_penalty 的 active_rate=99.8% (>95%)，signed_share=-11.9%（为负），符合全覆盖惩罚标记条件。
- **量级对比**（每步均值 = episode_sum_mean / len）：
  - 最大两个：progress_reward (0.72)、contact_reward (0.15)
  - 最小两个：settlement_penalty (0.00)、landing_speed_penalty (0.12)（按绝对值）
- 无“主导”组件（magnitude_share 最大值 73.2%，未突破 80%）
- 无“死组件”（最低 active_rate 为 7.8%，未低于 2%）
- 无“淹没”组件（active_rate>90% 的组件中仅 landing_speed_penalty，但其 magnitude_share=11.9%，未低于 3%）
- 无修改变化（仅有单轮历史，无法计算 Δ）