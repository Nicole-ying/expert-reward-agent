# 训练数据报告

## 1. 行为事实
- 得分：`score=21.452647`， range=[-121.813880, 228.013400]  
- 终止方式：terminated 9/20（成功着陆），truncated 11/20（超时）  
- 回合长度：`len=864.550`  
- Checkpoint 轨迹（官方得分）：  
  cp1=-173.8 → cp2=-44.9 → cp3=-44.1 → cp4=-53.1 → cp5=-116.6 → cp6=-42.6 → cp7=-69.1 → cp8=-144.2 → cp9=-117.6 → cp10=-31.4  
  整体趋势：早期剧烈波动，中后期震荡下降，最终在训练末段快速回升但仍为负值（-31.4），整体呈“V形”波动，未稳定上升。

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | upright_penalty_ep_sum | upright_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1  | -76.56 | 975.15 | terminated 1/20, truncated 19/20 | 0.000 | 0.0% | 0.635 | 100.0% | -3.714 | 100.0% | -12.561 | 100.0% |
| 2  | 63.72  | 668.05 | terminated 17/20, truncated 3/20 | 1075.630 | 5.2% | 1.835 | 98.8% | -7.301 | 100.0% | -9.517 | 99.0% |
| 3  | 21.45  | 864.55 | terminated 9/20, truncated 11/20 | 1011.468 | 8.2% | 1.769 | 99.9% | -11.452 | 100.0% | -9.103 | 99.9% |

## 3. 数字标记
- **主导**：`contact_bonus` magnitude_share = 97.7% > 80%，主导整体奖励量级。
- **死组件**：无（所有组件 active_rate ≥ 8.2%）。
- **淹没**：无（最高 active_rate 为 100%，但对应组件的 magnitude_share 均 > 1%，未出现淹没）。
- **全覆盖惩罚**：  
  - `upright_penalty` active_rate = 100.0%，signed_share = -1.1% → 满足全覆盖惩罚条件。  
  - `velocity_penalty` active_rate = 99.9%，signed_share = -0.9% → 满足全覆盖惩罚条件。
- **量级对比（每步均值 ep_sum/len）**：  
  - 最大：`contact_bonus` = 1.170，`goal_proximity` = 0.00205  
  - 最小：`upright_penalty` = -0.0132，`velocity_penalty` = -0.0105
- **修改变化（Δ 相对于上一轮 iter2）**：  
  | 组件 | Δ ep_sum | Δ active_rate |
  |---|---|---|
  | contact_bonus | -64.162 | +3.0% |
  | goal_proximity | -0.066 | +1.1% |
  | upright_penalty | -4.151 | 0.0% |
  | velocity_penalty | +0.414 | +0.9% |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| 接近/进度信号 | goal_proximity (iter1) → goal_proximity (iter2) → goal_proximity (iter3) | 0.635 → 1.835 → 1.769 | 无 |
| 着陆/接触信号 | contact_bonus (iter1, 全零) → contact_bonus (iter2) → contact_bonus (iter3) | 0.000 → 1075.630 → 1011.468 | 无 |
| 稳定性/约束信号 | upright_penalty + velocity_penalty (iter1) → upright_penalty + velocity_penalty (iter2) → upright_penalty + velocity_penalty (iter3) | -16.275 → -16.818 → -20.555 | 无 |

- 稳定性/约束信号组合中，velocity_penalty 从无门控（iter1）切换为高度门控（iter2+），但该设计在 iter2 后无反复；upright_penalty 始终保持全高度无门控。整体未出现方向振荡。