## 1. 行为事实
- 得分：score=-133.267334，range=[-214.122, -3.501409]
- 终止方式：terminated 18/20（成功），truncated 2/20（时间上限）
- 回合长度：len=656.0
- Checkpoint 轨迹：评估数据未提供（表中无可用的 score_mean/score_std）

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -85.93 | 71.10 | 20/0 | 4.254 | 2.2% | -0.785 | 22.1% | 0.539 | 100.0% | -2.748 | 23.9% |
| 2 | -104.33 | 301.40 | 19/1 | 3.128 | 0.3% | -11.463 | 76.1% | 2.207 | 100.0% | -1.587 | 5.1% |
| 3 | 91.82 | 900.70 | 5/15 | 1029.163 | 94.4% | -45.035 | 100.0% | 6.361 | 100.0% | -0.634 | 85.2% |
| 4 | 13.47 | 989.50 | 3/17 | 614.993 | 93.9% | -48.768 | 98.6% | 24.984 | 99.8% | -0.433 | 83.3% |
| 5 | -39.29 | 863.40 | 11/9 | 2866.351 | 91.8% | -39.860 | 92.3% | 4.049 | 99.8% | -2.026 | 76.7% |
| 6 | -20.44 | 611.10 | 14/6 | 3244.916 | 13.7% | -23.950 | 78.4% | 4.204 | 100.0% | -2.135 | 67.1% |
| 7 | -133.27 | 656.00 | 18/2 | 561.361 | 1.1% | -22.590 | 68.9% | 2.059 | 100.0% | -1.352 | 83.2% |

## 3. 数字标记
- **主导**：contact_bonus（magnitude_share 94.6% > 80%）
- **死组件**：contact_bonus（active_rate 1.1% < 2%）
- **淹没**：progress（active_rate 100% > 90%，magnitude_share 1.4% < 3%）
- **全覆盖惩罚**：无（无组件满足 active_rate > 95% 且 signed_share 为负）
- **量级对比**：
  - 每步均值最大两个：contact_bonus ≈ 0.856，progress ≈ 0.00314
  - 每步均值最小两个：fuel_penalty ≈ -0.0344，orientation_penalty ≈ -0.00206
- **修改变化（iter 6 → iter 7）**：
  | 组件 | Δ ep_sum | Δ active_rate |
  |---|---|---|
  | contact_bonus | -2683.55 | -12.6 pp |
  | fuel_penalty | +1.36 | -9.5 pp |
  | progress | -2.15 | 0.0 pp |
  | orientation_penalty | +0.78 | +16.1 pp |