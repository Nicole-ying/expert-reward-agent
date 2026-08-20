# 训练数据报告

## 1. 行为事实

- **得分**：score = –54.72，range = [–780.97, 8.90]
- **终止方式**：terminated = 20/20（全部坠毁/终止），truncated = 0/20（无时间截断）
- **回合长度**：len = 162.35 步
- **Checkpoint 轨迹**：本轮未提供 checkpoint 评估数据（无数据）

**各组件在固定评估中的行为数值**  
episode_sum_mean（ep_sum）与 active_rate：

| component | ep_sum | active_rate |
|---|---|---|
| contact_bonus | 174.654 | 33.7% |
| progress | 4.250 | 100.0% |
| fuel_penalty | –4.930 | 60.7% |
| orientation_penalty | –0.542 | 17.3% |

## 2. 骨架演进表

组件顺序统一为：A = contact_bonus, B = progress, C = fuel_penalty, D = orientation_penalty。  
“term”列格式为 `terminated/truncated`。

| iter | score | len | term | compA_ep_sum | compA_active | compB_ep_sum | compB_active | compC_ep_sum | compC_active | compD_ep_sum | compD_active |
|------|-------|------|------|---------------|--------------|---------------|--------------|---------------|--------------|---------------|--------------|
| 1 | -85.93 | 71.10 | 20/0 | 4.254 | 2.2% | 0.539 | 100.0% | -0.785 | 22.1% | -2.748 | 23.9% |
| 2 | -104.33 | 301.40 | 19/1 | 3.128 | 0.3% | 2.207 | 100.0% | -11.463 | 76.1% | -1.587 | 5.1% |
| 3 | 91.82 | 900.70 | 5/15 | 1029.163 | 94.4% | 6.361 | 100.0% | -45.035 | 100.0% | -0.634 | 85.2% |
| 4 | 13.47 | 989.50 | 3/17 | 614.993 | 93.9% | 24.984 | 99.8% | -48.768 | 98.6% | -0.433 | 83.3% |
| 5 | -39.29 | 863.40 | 11/9 | 2866.351 | 91.8% | 4.049 | 99.8% | -39.860 | 92.3% | -2.026 | 76.7% |
| 6 | -20.44 | 611.10 | 14/6 | 3244.916 | 13.7% | 4.204 | 100.0% | -23.950 | 78.4% | -2.135 | 67.1% |
| 7 | -133.27 | 656.00 | 18/2 | 561.361 | 1.1% | 2.059 | 100.0% | -22.590 | 68.9% | -1.352 | 83.2% |
| 8 (当前) | -54.72 | 162.35 | 20/0 | 174.654 | 33.7% | 4.250 | 100.0% | -4.930 | 60.7% | -0.542 | 17.3% |

## 3. 数字标记

（以下仅基于 **本轮 iter_08** 的评估数据，与 iter_07 对比的变化部分使用上一轮数据）

- **主导**：contact_bonus 的 magnitude_share = 92.7%（>80%），属于主导组件。
- **死组件**：无（所有组件 active_rate 均 ≥ 2%）。
- **淹没**：无（active_rate > 90% 的组件 progress、fuel_penalty、orientation_penalty 的 magnitude_share 分别为 4.4%、2.6%、0.3%，其中 progress 的 magnitude_share 未低于 3%，另两者 active_rate 未超过 90%）。
- **全覆盖惩罚**：无（active_rate > 95% 的组件 progress 的 signed_share 为 +2.3%，不为负）。
- **量级对比**（每步均值 = ep_sum ÷ len）：  
  最大两个：contact_bonus (≈1.0756) > progress (≈0.0262)  
  最小两个：orientation_penalty (≈–0.00334) > fuel_penalty (≈–0.03036)  
  （按 signed 值排序）
- **修改变化**（iter_08 相对于 iter_07 的差分）：
  - contact_bonus: Δep_sum = –386.707, Δactive_rate = +32.6 百分点
  - progress: Δep_sum = +2.191, Δactive_rate = 0 百分点
  - fuel_penalty: Δep_sum = +17.660, Δactive_rate = –8.2 百分点
  - orientation_penalty: Δep_sum = +0.810, Δactive_rate = –65.9 百分点