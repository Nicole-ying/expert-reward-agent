## 1. 行为事实
- 得分：score=-183.32，range=[-284.46, -99.25]
- 终止方式：terminated 7/20，truncated 13/20
- 回合长度：len=947.35
- Checkpoint 轨迹：未提供（无数据）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | contact_quality_ep_sum | contact_quality_active | success_proxy_ep_sum | success_proxy_active | thrust_penalty_ep_sum | thrust_penalty_active |
|------|--------|------|------|------------------------|-------------------------|--------------------------|---------------------------|-----------------------|------------------------|-------------------------|---------------------------|-----------------------|------------------------|-------------------------|--------------------------|
| 1    | -121.00 | 68.35 | 20/0 | -148.505 | 100.0% | -93.226 | 99.9% | -43.986 | 100.0% | 0.766 | 0.2% | 1.432 | 0.1% | -0.215 | 6.3% |
| 2    | -183.32 | 947.35 | 7/13 | -3768.393 | 100.0% | -24.150 | 100.0% | -9.950 | 100.0% | 0.000 | 0.0% | 5792.231 | 100.0% | -47.368 | 100.0% |

## 3. 数字标记
- **死组件**：contact_quality（active_rate=0.0% < 2%）
- **淹没**：thrust_penalty（active_rate=100.0% > 90%，magnitude_share=0.5% < 3%）、velocity_penalty（active_rate=100.0%，magnitude_share=0.3%）、angle_penalty（active_rate=100.0%，magnitude_share=0.1%）
- **全覆盖惩罚**：goal_proximity（active_rate=100.0% > 95%，signed_share=-39.1%）、thrust_penalty（active_rate=100.0%，signed_share=-0.5%）、velocity_penalty（active_rate=100.0%，signed_share=-0.3%）、angle_penalty（active_rate=100.0%，signed_share=-0.1%）
- **量级对比**（每步均值 = episode_sum_mean ÷ len）  
  - 最大值（绝对值）：success_proxy = 6.114，goal_proximity = 3.977  
  - 最小值（绝对值）：contact_quality = 0.000，angle_penalty = 0.0105
- **修改变化（iter1→iter2）**  
  - episode_sum_mean Δ：goal_proximity -3619.888，velocity_penalty +69.076，angle_penalty +34.036，contact_quality -0.766，success_proxy +5790.799，thrust_penalty -47.153  
  - active_rate Δ：goal_proximity 0.0，velocity_penalty +0.1，angle_penalty 0.0，contact_quality -0.2，success_proxy +99.9，thrust_penalty +93.7