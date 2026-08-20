## 1. 行为事实
- 得分：score=-120.998945，range=[-150.975402, -99.021460]
- 终止方式：terminated 20/20，truncated 0/20（全部为提前终止，无时间截断）
- 回合长度：len=68.35
- Checkpoint 轨迹：无数据

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | success_proxy_ep_sum | success_proxy_active | contact_quality_ep_sum | contact_quality_active | thrust_penalty_ep_sum | thrust_penalty_active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | -120.999 | 68.35 | 20T/0Tr | -148.505 | 100.0% | -93.226 | 99.9% | -43.986 | 100.0% | 1.432 | 0.1% | 0.766 | 0.2% | -0.215 | 6.3% |

## 3. 数字标记
- **死组件**：success_proxy（active_rate=0.1%），contact_quality（active_rate=0.2%）
- **全覆盖惩罚**：goal_proximity（active_rate=100.0%，signed_share=-51.5%），velocity_penalty（active_rate=99.9%，signed_share=-32.4%），angle_penalty（active_rate=100.0%，signed_share=-15.3%）
- **量级对比**（每步均值）：最大两个组件 success_proxy (0.0210)、contact_quality (0.0112)；最小两个组件 goal_proximity (-2.17)、velocity_penalty (-1.36)
- **修改变化**：无上一轮