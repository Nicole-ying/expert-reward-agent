## 1. 行为事实
- 得分：score=203.999462，range=[68.170546, 245.278434]
- 终止方式：terminated 18/20，truncated 2/20
- 回合长度：len=569.650000
- Checkpoint 轨迹：无数据

## 2. 骨架演进表
| iter | score | len | term | contact_completion_ep_sum | contact_completion_active | soft_landing_speed_penalty_ep_sum | soft_landing_speed_penalty_active | approach_progress_ep_sum | approach_progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 81.184393 | 890.750000 | 8t/12tr | 478.927441 | 15.8% | -73.990109 | 100.0% | 5.892979 | 100.0% | -5.213019 | 100.0% |
| 2 | 215.504114 | 517.900000 | 18t/2tr | 356.972250 | 18.4% | -60.296305 | 99.8% | 6.596181 | 97.7% | -2.671592 | 100.0% |
| 3 | 203.999462 | 569.650000 | 18t/2tr | 363.115059 | 16.5% | -71.365811 | 99.8% | 19.318184 | 97.8% | -3.520726 | 100.0% |

## 3. 数字标记
- **主导**：无（最大 magnitude_share 为 79.1%，未超过 80%）
- **死组件**：无（所有组件 active_rate > 2%）
- **淹没**：orientation_penalty（active_rate=100.0%，magnitude_share=0.8%<3%）
- **全覆盖惩罚**：soft_landing_speed_penalty（active_rate=99.8%，signed_share=-15.6%）；orientation_penalty（active_rate=100.0%，signed_share=-0.8%）
- **量级对比（每步均值大小）**：
  - 最大两个：contact_completion (0.6374)，soft_landing_speed_penalty (0.1253，绝对值)
  - 最小两个：orientation_penalty (0.0062，绝对值)，approach_progress (0.0339)
- **修改变化（iter_02 → iter_03）**：
  - contact_completion：ep_sum Δ=+6.14，active_rate Δ=-1.9%
  - soft_landing_speed_penalty：ep_sum Δ=-11.07，active_rate Δ=0.0%
  - approach_progress：ep_sum Δ=+12.72，active_rate Δ=+0.1%
  - orientation_penalty：ep_sum Δ=-0.85，active_rate Δ=0.0%