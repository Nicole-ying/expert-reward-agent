## 1. 行为事实
- 得分：score=26.45，range=[0.55, 50.01]
- 终止方式：terminated 20/20（全部成功终止），truncated 0/20
- 回合长度：len=206.50
- Checkpoint 轨迹：cp40%=0.00 ± 0.00，cp80%=0.00 ± 0.00，轨迹呈水平直线，无上升/下降趋势

## 2. 骨架演进表
| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | crash_penalty_ep_sum | crash_penalty_active | progress_ep_sum | progress_active | vel_damping_ep_sum | vel_damping_active | angle_stab_ep_sum | angle_stab_active | fuel_penalty_ep_sum | fuel_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -155.98 | 154.40 | 20/20 | 1.5822 | 100% | -1.8040 | 58.4% | -1.1405 | 100% | -0.0668 | 5.3% | — | — | — | — | — | — | — | — | — | — | — | — |
| 2 | -119.51 | 68.30 | 20/20 | 5.5835 | 100% | -0.0710 | 5.2% | -0.2049 | 100% | — | — | 5.0000 | 0.1% | — | — | — | — | — | — | — | — | — | — |
| 3 | -203.45 | 154.15 | 20/20 | 1.4876 | 100% | -1.4720 | 47.7% | -2.4599 | 100% | — | — | 0.0000 | 0.0% | 0.0000 | 0.0% | — | — | — | — | — | — | — | — |
| 4 | -114.38 | 68.45 | 20/20 | 5.5969 | 100% | -0.0970 | 7.1% | -0.0866 | 100% | — | — | 30.0000 | 0.4% | — | — | — | — | — | — | — | — | — | — |
| 5 | -46.51 | 740.15 | 7/20 (13 tr.) | 5.1046 | 99.9% | -13.4690 | 91.0% | -6.8280 | 100% | — | — | 1195.0000 | 1.6% | — | — | — | — | — | — | — | — | — | — |
| 6 | -111.18 | 68.45 | 20/20 | 5.6029 | 100% | -0.0670 | 4.9% | -0.0200 | 100% | — | — | 0.0000 | 0.0% | — | — | — | — | — | — | — | — | — | — |
| 7 | -98.93 | 69.60 | 20/20 | 5.5530 | 100% | -0.1780 | 12.8% | -0.2209 | 100% | — | — | 5.0000 | 0.1% | — | — | — | — | — | — | — | — | — | — |
| 8 | 26.45 | 206.50 | 20/20 | — | — | — | — | — | — | — | — | — | — | — | — | 2.6669 | 100% | -1.5767 | 99.9% | -0.2635 | 100% | -1.2315 | 59.6% |

## 3. 数字标记
- **主导**：无（最大 magnitude_share = 47.5%）
- **死组件**：无（最低 active_rate = 59.6%）
- **淹没**：无（无组件同时满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：vel_damping（active_rate=99.9%, signed_share=-27.0%），angle_stab（active_rate=100%, signed_share=-4.5%）
- **每步均值量级排序**（ep_sum / len）  
  最大两个（magnitude）：progress (+0.0129)，vel_damping (−0.0076)  
  最小两个（magnitude）：angle_stab (−0.0013)，fuel_penalty (−0.0060)
- **修改变化**（相较 iter_07）：组件集完全重构，无同名组件，Δ 不适用

## 4. 语义演化

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| 接近/进度信号 | progress_shaping → progress_shaping → progress_shaping → progress_shaping → progress_shaping → progress_shaping → progress_shaping → progress | -155.98 → -119.51 → -203.45 → -114.38 → -46.51 → -111.18 → -98.93 → 26.45 | 无切换（始终为连续奖励、状态改善量） |
| 着陆/接触信号 | landing_velocity_penalty → success_landing_bonus → success_landing_bonus, crash_penalty → success_landing_bonus → success_landing_bonus → success_landing_bonus → success_landing_bonus → (无) | 同上 | 稀疏bonus↔连续奖励：1次切换（v1连续惩罚 → v2稀疏bonus） |
| 稳定性/约束信号 | angle_penalty → angle_penalty → angle_penalty → angle_penalty → angle_penalty → angle_penalty → angle_penalty → angle_stab + vel_damping | 同上 | 无切换（始终为连续全高度惩罚） |
| 效率/动作代价 | action_cost → action_cost → action_cost → action_cost → action_cost → action_cost → action_cost → fuel_penalty | 同上 | 无切换（始终为连续惩罚） |