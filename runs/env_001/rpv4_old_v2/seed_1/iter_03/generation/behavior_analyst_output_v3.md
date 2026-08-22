# 训练数据报告

## 1. 行为事实
- 得分：score=-104.33，range=[-180.87, 31.48]
- 终止方式：terminated 19/20（成功/失败），truncated 1/20（时间上限）
- 回合长度：len=301.40
- Checkpoint 轨迹：无 checkpoint 数据（本次评估未提供官方得分曲线）

## 2. 骨架演进表

| iter | score | len | term | progress_ep_sum | progress_active | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active |
|------|--------|------|------|-----------------|-----------------|----------------------|---------------------|---------------------|---------------------|----------------------------|----------------------------|
| 1 | -85.93 | 71.10 | 20/20 (0 truncated) | 0.539 | 100.0% | 4.254 | 2.2% | -0.785 | 22.1% | -2.748 | 23.9% |
| 2 | -104.33 | 301.40 | 19/20 (1 truncated) | 2.207 | 100.0% | 3.128 | 0.3% | -11.463 | 76.1% | -1.587 | 5.1% |

说明：
- 所有数值直接取自 `episode_sum_mean`，`active_rate` 为百分比。
- iter 1 为初始版本（历史最佳，score=-85.93），iter 2 为本轮。

## 3. 数字标记

### 主导
- 无组件 magnitude_share > 80%（本轮最高 fuel_penalty 50.9%）。

### 死组件
- **contact_bonus**：active_rate = 0.3% (< 2%)

### 淹没
- 无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。

### 全覆盖惩罚
- 无组件同时满足 active_rate > 95% 且 signed_share 为负。

### 量级对比（每步均值 = ep_sum / len，取两位小数）
- 总 len=301.40
  - fuel_penalty: -11.463 / 301.40 = -0.0380
  - orientation_penalty: -1.587 / 301.40 = -0.0053
  - progress: 2.207 / 301.40 = 0.0073
  - contact_bonus: 3.128 / 301.40 = 0.0104

**最大的两个**（按数值大小）：
1. contact_bonus: 0.0104
2. progress: 0.0073

**最小的两个**（按数值大小）：
1. fuel_penalty: -0.0380
2. orientation_penalty: -0.0053

### 修改变化（iter 1 → iter 2）
| 组件 | ep_sum Δ | active_rate Δ |
|------|----------|----------------|
| progress | +1.668 | 0% |
| contact_bonus | -1.125 | -1.9% |
| fuel_penalty | -10.678 | +54.0% |
| orientation_penalty | +1.161 | -18.8% |