# 训练数据报告

## 1. 行为事实
- 得分：score = -39.288，range = [-232.389, 154.021]
- 终止方式：terminated 11/20（成功着陆），truncated 9/20（超时）
- 回合长度：len = 863.4
- Checkpoint 轨迹：无数据

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|-----|--------|------|------|---------------------|---------------------|--------------------|--------------------|----------------|---------------|---------------------------|--------------------------|
| 1 | -85.93 | 71.10 | 20/0 | 4.25 | 2.2% | -0.79 | 22.1% | 0.54 | 100.0% | -2.75 | 23.9% |
| 2 | -104.33 | 301.40 | 19/1 | 3.13 | 0.3% | -11.46 | 76.1% | 2.21 | 100.0% | -1.59 | 5.1% |
| 3 | 91.82 | 900.70 | 5/15 | 1029.16 | 94.4% | -45.04 | 100.0% | 6.36 | 100.0% | -0.63 | 85.2% |
| 4 | 13.47 | 989.50 | 3/17 | 614.99 | 93.9% | -48.77 | 98.6% | 24.98 | 99.8% | -0.43 | 83.3% |
| 5 | -39.29 | 863.40 | 11/9 | 2866.35 | 91.8% | -39.86 | 92.3% | 4.05 | 99.8% | -2.03 | 76.7% |

## 3. 数字标记

- **主导**：contact_bonus 的 magnitude_share 为 98.3% (>80%)
- **死组件**：无（所有组件 active_rate > 2%）
- **淹没**：
  - progress：active_rate=99.8%，magnitude_share=0.2% (<3%)
  - fuel_penalty：active_rate=92.3%，magnitude_share=1.4% (<3%)
- **全覆盖惩罚**：无（无组件 active_rate > 95% 且 signed_share 为负）
- **量级对比（每步均值）**：
  - 最大两个：contact_bonus (3.320), progress (0.00469)
  - 最小两个：fuel_penalty (-0.04616), orientation_penalty (-0.00235)
- **修改变化（iter4 → iter5）**：
  - contact_bonus：ep_sum Δ = +2251.36，active_rate Δ = -2.1%（下降）
  - fuel_penalty：ep_sum Δ = +8.91（负值减小），active_rate Δ = -6.3%（下降）
  - progress：ep_sum Δ = -20.93，active_rate Δ = 0.0%
  - orientation_penalty：ep_sum Δ = -1.60（更负），active_rate Δ = -6.6%（下降）