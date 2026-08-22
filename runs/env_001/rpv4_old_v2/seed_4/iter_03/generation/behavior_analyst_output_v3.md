## 1. 行为事实
- 得分：score=215.50，range=[79.06, 258.14]
- 终止方式：terminated 18/20（成功），truncated 2/20（时间上限）
- 回合长度：len=517.90
- Checkpoint 轨迹：未提供数据（无10%步数得分轨迹）

## 2. 骨架演进表
| iter | score | len | term | contact_completion_ep_sum | contact_completion_active | soft_landing_speed_penalty_ep_sum | soft_landing_speed_penalty_active | approach_progress_ep_sum | approach_progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 81.18 | 890.75 | 8/12 | 478.93 | 15.8% | -73.99 | 100.0% | 5.89 | 100.0% | -5.21 | 100.0% |
| 2 | 215.50 | 517.90 | 18/2 | 356.97 | 18.4% | -60.30 | 99.8% | 6.60 | 97.7% | -2.67 | 100.0% |

## 3. 数字标记
- **主导**：`contact_completion` 的 magnitude_share=83.6% > 80%，占主导地位。
- **死组件**：无（所有组件 active_rate ≥ 18.4%）。
- **淹没**：`approach_progress`（active_rate=97.7%，magnitude_share=1.6%）和 `orientation_penalty`（active_rate=100.0%，magnitude_share=0.6%）的 active_rate > 90% 但 magnitude_share < 3%，属于淹没组件。
- **全覆盖惩罚**：`soft_landing_speed_penalty`（active_rate=99.8%，signed_share=-14.1%）和 `orientation_penalty`（active_rate=100.0%，signed_share=-0.6%）active_rate > 95% 且 signed_share 为负。
- **量级对比**（每步均值 = episode_sum_mean ÷ len）：
  - 最大两组件：`contact_completion` (+0.689)、`soft_landing_speed_penalty` (-0.116)
  - 最小两组件：`orientation_penalty` (-0.00516)、`approach_progress` (+0.0127)
- **修改变化**（iter_01 → iter_02）：
  - `contact_completion`：ep_sum Δ = -121.96，active_rate Δ = +2.6个百分点
  - `soft_landing_speed_penalty`：ep_sum Δ = +13.69，active_rate Δ = -0.2个百分点
  - `approach_progress`：ep_sum Δ = +0.70，active_rate Δ = -2.3个百分点
  - `orientation_penalty`：ep_sum Δ = +2.54，active_rate Δ = 0个百分点