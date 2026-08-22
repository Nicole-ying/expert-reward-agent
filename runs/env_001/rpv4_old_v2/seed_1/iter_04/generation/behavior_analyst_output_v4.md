## 1. 行为事实

- 得分：score=91.82，episode 得分范围 [-8.60, 239.53]
- 终止方式：terminated 5/20（成功接触），truncated 15/20（达到时间上限）
- 回合长度：len=900.70 步
- Checkpoint 轨迹：本轮未提供每 10% 步数的官方得分数据

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -85.93 | 71.1 | 20/0 | 4.25 | 2.2% | -0.78 | 22.1% | 0.54 | 100% | -2.75 | 23.9% |
| 2 | -104.33 | 301.4 | 19/1 | 3.13 | 0.3% | -11.46 | 76.1% | 2.21 | 100% | -1.59 | 5.1% |
| 3 | 91.82 | 900.7 | 5/15 | 1029.16 | 94.4% | -45.04 | 100% | 6.36 | 100% | -0.63 | 85.2% |

> term 列：terminated/truncated 计数（共 20 个固定评估 episode）  
> ep_sum = episode_sum_mean，active = active_rate

## 3. 数字标记

- **主导**：contact_bonus（magnitude_share=95.1% > 80%）
- **死组件**：无
- **淹没**：progress（active_rate=100% > 90%，magnitude_share=0.7% < 3%）
- **全覆盖惩罚**：fuel_penalty（active_rate=100% > 95%，signed_share=-4.2% 为负）
- **量级对比**（每步均值 = episode_sum_mean ÷ len，按绝对值排序）：  
  最大两个：contact_bonus（≈1.14）与 fuel_penalty（≈0.05）  
  最小两个：progress（≈0.007）与 orientation_penalty（≈0.0007）
- **修改变化**（iter02 → iter03，ep_sum 与 active_rate 的 Δ）  
  - contact_bonus：ep_sum +1026.04，active_rate +94.1%  
  - fuel_penalty：ep_sum -33.57（惩罚加深），active_rate +23.9%  
  - progress：ep_sum +4.15，active_rate 0%  
  - orientation_penalty：ep_sum +0.95（惩罚减轻），active_rate +80.1%