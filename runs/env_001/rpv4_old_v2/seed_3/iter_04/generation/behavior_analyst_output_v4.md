# 训练数据报告

## 1. 行为事实
- 评估得分：110.436884（范围 -51.736246 ~ 219.059345）
- 终止方式：成功着陆 (terminated) 18/20，时间耗尽 (truncated) 2/20
- 回合平均长度：528.55 步
- 无过早终止 (early_terminal) 事件（0/20）
- 无 reward 计算错误（reward_errors=0）
- Checkpoint 评估数据：未提供，无法写出轨迹

## 2. 骨架演进表

| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | orientation_stability_reward_ep_sum | orientation_stability_reward_active | safe_contact_reward_ep_sum | safe_contact_reward_active | velocity_damping_reward_ep_sum | velocity_damping_reward_active | landing_proxy_reward_ep_sum | landing_proxy_reward_active | action_penalty_ep_sum | action_penalty_active |
|------|-------|-----|------|-------------------------|------------------------|-------------------------------------|-------------------------------------|----------------------------|----------------------------|-------------------------------|-------------------------------|-----------------------------|-----------------------------|-----------------------|-----------------------|
| 1    | 144.17 | 965.45 | T:1/20, Tr:19/20 | 1.986 | 100.0% | -3.952 | 100.0% | 312.464 | 65.2% | -1.291 | 100.0% | - | - | - | - |
| 2    | 124.27 | 604.20 | T:13/20, Tr:7/20 | 1.878 | 99.2% | -9.479 | 100.0% | 852.000 | 7.1% | -1.435 | 99.2% | - | - | - | - |
| 3    | 110.44 | 528.55 | T:18/20, Tr:2/20 | 3.644 | 99.8% | -15.551 | 100.0% | - | - | - | - | 359.975 | 24.5% | -15.683 | 98.9% |

（“-”表示该组件在本轮未定义）

## 3. 数字标记

### 主导
- `landing_proxy_reward` magnitude_share = 91.1% (>80%)，主导奖励构成。

### 死组件
- 无死组件（所有组件 active_rate ≥ 24.5%，无低于 2% 的情况）。

### 淹没
- `progress_reward` active_rate = 99.8% (>90%)，magnitude_share = 1.0% (<3%)，被淹没。

### 全覆盖惩罚
- `action_penalty` active_rate = 98.9% (>95%)，signed_share = -4.0%（负），标记为全覆盖惩罚。
- `orientation_stability_reward` active_rate = 100.0% (>95%)，signed_share = -3.9%（负），标记为全覆盖惩罚。

### 量级对比（每步均值 ep_sum/len）
- 最大两个：
  `landing_proxy_reward` ≈ 0.681  
  `progress_reward` ≈ 0.007
- 最小两个（最负）：
  `action_penalty` ≈ -0.0297  
  `orientation_stability_reward` ≈ -0.0294

### 修改变化（iter_2 → iter_3）
- 消失组件：
  `safe_contact_reward` ep_sum Δ = -852.000， active_rate Δ = -7.1 ppt  
  `velocity_damping_reward` ep_sum Δ = +1.435， active_rate Δ = -99.2 ppt
- 新增组件：
  `landing_proxy_reward` ep_sum Δ = +359.975， active_rate Δ = +24.5 ppt  
  `action_penalty` ep_sum Δ = -15.683， active_rate Δ = +98.9 ppt
- 保留组件变化：
  `progress_reward` ep_sum Δ = +1.766， active_rate Δ = +0.6 ppt  
  `orientation_stability_reward` ep_sum Δ = -6.072， active_rate Δ = 0 ppt