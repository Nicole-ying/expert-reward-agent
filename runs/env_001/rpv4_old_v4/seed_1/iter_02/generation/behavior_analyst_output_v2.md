# 数据报告

## 1. 行为事实
- 得分：score=-2.976939，range=[-54.501032, 152.393336]
- 终止方式：terminated=5/20，truncated=15/20（时间上限）
- 回合长度：len=813.350000
- Checkpoint 轨迹（官方得分）：cp10%=-72.4 → cp20%=-83.6 → cp30%=-4.2 → cp40%=-31.9 → cp50%=-28.9 → cp60%=104.8 → cp70%=138.8 → cp80%=164.9 → cp90%=146.2 → cp100%=122.7。趋势：前期快速负值，30%处骤升至上正，随后波动，中后期为正且维持 >100，最终得分较最佳点（cp80%）有所回落，整体从 -72.4 上升至 122.7。

## 2. 骨架演进表
| iter | score | len | term | proximity_progress_ep_sum | proximity_progress_active | soft_landing_bonus_ep_sum | soft_landing_bonus_active | attitude_penalty_ep_sum | attitude_penalty_active |
|------|-------|-----|------|---------------------------|---------------------------|---------------------------|---------------------------|------------------------|-------------------------|
| 1    | -2.98 | 813.35 | 5T/15Tr | 36.7427 | 99.9% | 8.0484 | 0.3% | -1.3484 | 100.0% |

## 3. 数字标记
- **主导**：proximity_progress，magnitude_share=82.9% (>80%)
- **死组件**：soft_landing_bonus，active_rate=0.3% (<2%)
- **淹没**：attitude_penalty，active_rate=100.0% (>90%) 且 magnitude_share=2.5% (<3%)
- **全覆盖惩罚**：attitude_penalty，active_rate=100.0% (>95%) 且 signed_share 为负 (-2.5%)
- **量级对比**（每步均值，ep_sum÷len）：
  - proximity_progress: 0.04517
  - soft_landing_bonus: 0.00989
  - attitude_penalty: 0.00166（绝对值）
  最大两个：proximity_progress (0.04517), soft_landing_bonus (0.00989)  
  最小两个：soft_landing_bonus (0.00989), attitude_penalty (0.00166)
- **修改变化**：无（首轮迭代）

## 4. 语义演化（跨轮组件功能追踪）
| 语义功能 | 各轮组件名 | 各轮 ep_sum_mean | 方向切换 |
|---------|----------|-----------------|---------|
| 接近/进度信号 | proximity_progress | 36.74 | 0次 |
| 着陆/接触信号 | soft_landing_bonus | 8.05 | 0次 |
| 稳定性/约束信号 | attitude_penalty | -1.35 | 0次 |