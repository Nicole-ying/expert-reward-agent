## 1. 行为事实
- 得分：score=-63.97，range=[-95.51, -20.66]（20 episodes）
- 终止方式：terminated 1/20（成功着陆），truncated 19/20（达到时间上限）；早停（<150步且得分<-50）0/20
- 回合长度：平均963.75步
- Checkpoint轨迹：无评估数据（反馈中为空）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | contact_quality_ep_sum | contact_quality_active | success_proxy_ep_sum | success_proxy_active | thrust_penalty_ep_sum | thrust_penalty_active | soft_success_ep_sum | soft_success_active | goal_distance_progress_ep_sum | goal_distance_progress_active | soft_landing_ep_sum | soft_landing_active | orientation_penalty_ep_sum | orientation_penalty_active | landing_contact_success_ep_sum | landing_contact_success_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -121.00 | 68.35 | 20/20,0/20 | -148.51 | 100.0% | -93.23 | 99.9% | -43.99 | 100.0% | 0.77 | 0.2% | 1.43 | 0.1% | -0.22 | 6.3% | - | - | - | - | - | - | - | - | - | - |
| 2 | -183.32 | 947.35 | 7/20,13/20 | -3768.39 | 100.0% | -24.15 | 100.0% | -9.95 | 100.0% | 0.00 | 0.0% | 5792.23 | 100.0% | -47.37 | 100.0% | - | - | - | - | - | - | - | - | - | - |
| 3 | -120.25 | 68.40 | 20/20,0/20 | -156.89 | 100.0% | -93.44 | 99.8% | -38.99 | 100.0% | 0.77 | 0.2% | 1.43 | 0.1% | -0.18 | 5.3% | - | - | - | - | - | - | - | - | - | - |
| 4 | -1471.59 | 435.40 | 20/20,0/20 | 163.93 | 100.0% | -376.90 | 100.0% | -3.79 | 100.0% | 0.00 | 0.0% | - | - | -21.77 | 100.0% | 12716.35 | 96.2% | - | - | - | - | - | - | - | - |
| 5 | -63.97 | 963.75 | 1/20,19/20 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | 3.78 | 100.0% | -4.93 | 100.0% | -1.22 | 100.0% | 0.00 | 0.0% |

## 3. 数字标记
- **主导**：无（所有组件的magnitude_share均未超过80%）
- **死组件**：landing_contact_success（active_rate=0.0%，<2%）
- **淹没**：无（所有active_rate>90%的组件magnitude_share≥8.9%，均>3%）
- **全覆盖惩罚**：soft_landing（active_rate=100%>95%且signed_share=-35.9%<0）、orientation_penalty（active_rate=100%>95%且signed_share=-8.9%<0）
- **量级对比（每步均值）**：
  - 最大两个：goal_distance_progress（0.00392）、landing_contact_success（0.0）
  - 最小两个：soft_landing（-0.00512）、orientation_penalty（-0.00126）
- **修改变化（与上一轮iter4比较）**：无共同奖励组件，Δ计算不适用