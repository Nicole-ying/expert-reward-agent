# 训练数据报告

## 1. 行为事实
- 得分：score=-15.01，range=[-43.54, 26.85]
- 终止方式：terminated 0/20，truncated 20/20（100% 时间上限）
- 回合长度：len=1000.0
- Checkpoint 轨迹：官方得分 cp10%=-261.9 → cp20%=-453.5 → cp30%=-145.7 → cp40%=-25.6 → cp50%=-11.9 → cp60%=-22.6 → cp70%=-18.2 → cp80%=193.2 → cp90%=-10.4 → cp100%=-19.9（先升后降，80% 步数出现孤立正峰后回落至负区间）

## 2. 骨架演进表
| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | contact_bonus_ep_sum | contact_bonus_active | descent_incentive_ep_sum | descent_incentive_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -32.35 | 1000.0 | 0T/20Tr | 3.758 | 100.0% | -0.486 | 100.0% | -3.959 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | 107.13 | 775.6 | 11T/9Tr | 3.963 | 99.1% | -5.313 | 99.2% | -3.090 | 100.0% | 154.465 | 2.8% | — | — |
| 3 | -26.17 | 1000.0 | 0T/20Tr | 7.685 | 100.0% | -4.758 | 100.0% | -4.038 | 100.0% | 0.000 | 0.0% | — | — |
| 4 | 183.52 | 710.25 | 19T/1Tr | 7.951 | 98.1% | -5.736 | 98.4% | -3.386 | 100.0% | 114.083 | 4.9% | 0.779 | 72.4% |
| 5 | -15.01 | 1000.0 | 0T/20Tr | 7.881 | 100.0% | -6.826 | 100.0% | -3.615 | 100.0% | 0.000 | 0.0% | 0.624 | 66.1% |

## 3. 数字标记
- **死组件**：contact_bonus（active_rate=0.0% < 2%）
- **全覆盖惩罚**：
  - velocity_penalty（active_rate=100% > 95%，signed_share=-35.1%）
  - orientation_penalty（active_rate=100% > 95%，signed_share=-18.6%）
- **量级对比**（每步均值 = ep_sum / 1000，按带符号数值排序）：
  - 最大值（正）前二：goal_proximity (+0.00788)、descent_incentive (+0.00062)
  - 最小值（负）前二：velocity_penalty (-0.00683)、orientation_penalty (-0.00361)
- **修改变化**（iter_5 相对于 iter_4）：
  | 组件 | Δ ep_sum | Δ active_rate |
  |---|---|---|
  | goal_proximity | -0.070 | +1.9% |
  | velocity_penalty | -1.090 | +1.6% |
  | orientation_penalty | -0.229 | 0.0% |
  | descent_incentive | -0.155 | -6.3% |
  | contact_bonus | -114.083 | -4.9% |

## 4. 语义演化（跨轮组件功能追踪）
| 语义功能 | 各轮组件名 | 各轮得分（ep_sum_mean） | 方向切换 |
|---|---|---|---|
| 接近/进度信号 | goal_proximity（iter1-5）<br>descent_incentive（iter4-5） | goal_proximity: 3.758 → 3.963 → 7.685 → 7.951 → 7.881<br>descent_incentive: — → — → — → 0.779 → 0.624 | 无（均为连续改善量奖励，无对立设计切换） |
| 着陆/接触信号 | contact_bonus（iter1-5） | 0.000 → 154.465 → 0.000 → 114.083 → 0.000 | 无（始终为稀疏事件奖励，无切换为连续形式） |
| 稳定性/约束信号 | velocity_penalty（iter1-5）<br>orientation_penalty（iter1-5） | velocity_penalty: -0.486 → -5.313 → -4.758 → -5.736 → -6.826<br>orientation_penalty: -3.959 → -3.090 → -4.038 → -3.386 → -3.615 | velocity_penalty：门控方式切换1次（iter1距离基门控→iter2及以后高度指数门控）<br>orientation_penalty：无切换（全高度固定激活） |