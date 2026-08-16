# 数据报告

## 1. 行为事实
- 得分：score=-32.35，range=[-63.84, 10.01]
- 终止方式：terminated 0/20（无成功/失败提前终止），truncated 20/20（全部达到1000步时间上限）
- 回合长度：len=1000.00
- Checkpoint 轨迹：官方得分从 cp10%=38.8 → cp20%=-27.4 → cp30%=-33.9 → cp40%=-30.5 → cp50%=-19.8 → cp60%=-50.7 → cp70%=-32.3 → cp80%=-35.5 → cp90%=-10.3 → cp100%=-27.8；整体从高正分骤降后维持在负值区间波动。

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active | contact_bonus_ep_sum | contact_bonus_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -32.35 | 1000.00 | terminated=0/20, truncated=20/20 | 3.758 | 100.0% | -3.959 | 100.0% | -0.486 | 100.0% | 0.000 | 0.0% |

## 3. 数字标记
- **死组件**：contact_bonus active_rate=0.0% (<2%)。
- **量级对比**：每步均值（ep_sum/len）绝对值最大的两个组件：orientation_penalty (0.003959)、goal_proximity (0.003758)；最小的两个组件：velocity_penalty (0.000486)、contact_bonus (0.0)。

（无主导、淹没、全覆盖惩罚，且无上一轮 Δ 数据）

## 4. 语义演化
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | goal_proximity | 3.758 | 无历史 |
| 着陆/接触信号 | contact_bonus | 0.000 | 无历史 |
| 稳定性/约束信号 | orientation_penalty, velocity_penalty | -3.959, -0.486 | 无历史 |