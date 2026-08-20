# 数据报告

## 1. 行为事实
- 得分：score=-85.93，range=[-136.39, -28.36]
- 终止方式：terminated 20/20（全部失败），truncated 0/20
- 回合长度：len=71.10
- Checkpoint 轨迹：无数据（pct 表为空）

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | orientation_penalty_ep_sum | orientation_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active |
|------|-------|-----|------|---------------------|---------------------|---------------------------|---------------------------|---------------------|---------------------|-----------------|-----------------|
| 1    | -85.93 | 71.10 | terminated=20/20, truncated=0/20 | 4.2538 | 2.2% | -2.7482 | 23.9% | -0.7850 | 22.1% | 0.5393 | 100.0% |

## 3. 数字标记
- **主导**：无（无组件 magnitude_share > 80%）
- **死组件**：无（active_rate 最低为 contact_bonus 2.2%，>2% 阈值）
- **淹没**：无（progress active_rate 100% 但 magnitude_share 6.8% > 3%）
- **全覆盖惩罚**：无（无组件满足 active_rate > 95% 且 signed_share < 0）
- **量级对比**（每步均值 = episode_sum_mean / len）：
  - 最大：contact_bonus 0.0598，progress 0.0076
  - 最小：orientation_penalty -0.0387，fuel_penalty -0.0110
- **修改变化**：无上一轮数据，Δ 为空