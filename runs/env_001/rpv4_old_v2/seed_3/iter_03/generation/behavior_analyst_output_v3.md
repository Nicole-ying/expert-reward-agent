## 1. 行为事实
- 得分：score=124.271173，range=[-87.308558, 244.895182]
- 终止方式：terminated 13/20，truncated 7/20
- 回合长度：len=604.20
- Checkpoint 轨迹：无数据（本轮未提供 checkpoint 评估）

## 2. 骨架演进表
| iter | score | len | term | safe_contact_ep_sum | safe_contact_active | orientation_stability_ep_sum | orientation_stability_active | progress_ep_sum | progress_active | velocity_damping_ep_sum | velocity_damping_active |
|------|--------|---------|------|---------------------|---------------------|------------------------------|------------------------------|-----------------|-----------------|-------------------------|-------------------------|
| 1    | 144.17 | 965.45  | 1/20 terminated, 19/20 truncated | 312.46              | 65.2%               | -3.95                        | 100.0%                       | 1.99            | 100.0%          | -1.29                   | 100.0%                  |
| 2    | 124.27 | 604.20  | 13/20 terminated, 7/20 truncated | 852.00              | 7.1%                | -9.48                        | 100.0%                       | 1.88            | 99.2%           | -1.44                   | 99.2%                   |

## 3. 数字标记
- **主导**：`safe_contact_reward` magnitude_share=98.5% > 80%，在奖励量级中占绝对主导。
- **淹没**：
  - `orientation_stability_reward` active_rate=100.0% > 90% 且 magnitude_share=1.1% < 3%
  - `progress_reward` active_rate=99.2% > 90% 且 magnitude_share=0.2% < 3%
  - `velocity_damping_reward` active_rate=99.2% > 90% 且 magnitude_share=0.2% < 3%
- **全覆盖惩罚**：
  - `orientation_stability_reward` active_rate=100.0% > 95% 且 signed_share=-1.1% （负）
  - `velocity_damping_reward` active_rate=99.2% > 95% 且 signed_share=-0.2% （负）
- **量级对比**（每步均值 = ep_sum ÷ len，按绝对值）：
  - 最大两个组件：`safe_contact_reward`（1.410）、`orientation_stability_reward`（0.0157）
  - 最小两个组件：`velocity_damping_reward`（0.00238）、`progress_reward`（0.00311）
- **修改变化**（iter=2 − iter=1）：
  - `safe_contact_reward`：ep_sum +539.54，active_rate −58.1 pp
  - `orientation_stability_reward`：ep_sum −5.53，active_rate 0.0 pp
  - `progress_reward`：ep_sum −0.108，active_rate −0.8 pp
  - `velocity_damping_reward`：ep_sum −0.145，active_rate −0.8 pp