## 1. 行为事实
- 得分：score=144.17，range=[112.27, 280.37]
- 终止方式：terminated 1/20，truncated 19/20
- 回合长度：len=965.45
- Checkpoint 轨迹：无评估数据

## 2. 骨架演进表
| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | velocity_damping_reward_ep_sum | velocity_damping_reward_active | orientation_stability_reward_ep_sum | orientation_stability_reward_active | safe_contact_reward_ep_sum | safe_contact_reward_active |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144.17 | 965.45 | 1/20 terminated, 19/20 truncated | 1.986 | 100.0% | -1.291 | 100.0% | -3.952 | 100.0% | 312.464 | 65.2% |

## 3. 数字标记
- **主导**：safe_contact_reward 的 magnitude_share=97.7%，超过 80%。
- **死组件**：无（所有组件 active_rate 均 ≥ 65.2%）。
- **淹没**：
  - progress_reward：active_rate=100.0%，magnitude_share=0.6%
  - velocity_damping_reward：active_rate=100.0%，magnitude_share=0.4%
  - orientation_stability_reward：active_rate=100.0%，magnitude_share=1.2%
- **全覆盖惩罚**：
  - orientation_stability_reward：active_rate=100.0%，signed_share 为负（-1.2%）
  - velocity_damping_reward：active_rate=100.0%，signed_share 为负（-0.4%）
- **量级对比**（ep_sum ÷ len 从大到小排序）：
  - 最大两个：safe_contact_reward (0.3236/步)，progress_reward (0.00206/步)
  - 最小两个：velocity_damping_reward (-0.00134/步)，orientation_stability_reward (-0.00409/步)
- **修改变化**：无上一轮数据，无法计算 Δ。