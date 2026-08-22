## 1. 行为事实
- 得分：score=81.18，range=[-27.61, 204.85]
- 终止方式：terminated 8/20（成功），truncated 12/20（时间上限）
- 回合长度：len=890.75
- Checkpoint 轨迹：（未提供数据）

## 2. 骨架演进表
| iter | score | len | term | approach_progress_ep_sum | approach_progress_active | soft_landing_speed_penalty_ep_sum | soft_landing_speed_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | contact_completion_ep_sum | contact_completion_active |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 81.18 | 890.75 | 8T/12Tr | 5.89 | 1.00 | -73.99 | 1.00 | -5.21 | 1.00 | 478.93 | 0.158 |

## 3. 数字标记
- **主导**：contact_completion（magnitude_share=84.8% > 80%）
- **死组件**：无（所有 active_rate ≥ 15.8%）
- **淹没**：approach_progress（active_rate=100%, magnitude_share=1.2% < 3%）、orientation_penalty（active_rate=100%, magnitude_share=0.9% < 3%）
- **全覆盖惩罚**：soft_landing_speed_penalty（active_rate=100%, signed_share=-13.1%）、orientation_penalty（active_rate=100%, signed_share=-0.9%）
- **量级对比**（每步均值 ep_sum / len）：
  - 最大两个：contact_completion 0.538，approach_progress 0.0066
  - 最小两个：soft_landing_speed_penalty -0.0831，orientation_penalty -0.00585
- **修改变化**：无（首轮，无历史对比）