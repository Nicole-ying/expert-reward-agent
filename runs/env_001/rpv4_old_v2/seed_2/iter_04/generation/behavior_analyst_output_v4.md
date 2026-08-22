## 1. 行为事实

- 得分：score = -120.253372，range = [-151.335058, -98.281266]
- 终止方式：terminated 20/20（全部提前终止），truncated 0/20
- 回合长度：len = 68.4
- Checkpoint 轨迹：本轮未提供 checkpoint 评估数据（表中为空）

## 2. 骨架演进表

| iter | score | len | term | goal_ep_sum | goal_active | vel_ep_sum | vel_active | ang_ep_sum | ang_active | contact_ep_sum | contact_active | proxy_ep_sum | proxy_active | thrust_ep_sum | thrust_active |
|:---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -120.998945 | 68.35 | T=20/20, TR=0/20 | -148.505275 | 100.0% | -93.225748 | 99.9% | -43.986293 | 100.0% | 0.766166 | 0.2% | 1.432261 | 0.1% | -0.215 | 6.3% |
| 2 | -183.316556 | 947.35 | T=7/20, TR=13/20 | -3768.392629 | 100.0% | -24.149682 | 100.0% | -9.949903 | 100.0% | 0.0 | 0.0% | 5792.231178 | 100.0% | -47.3675 | 100.0% |
| 3 | -120.253372 | 68.40 | T=20/20, TR=0/20 | -156.891027 | 100.0% | -93.439696 | 99.8% | -38.994126 | 100.0% | 0.768669 | 0.2% | 1.434087 | 0.1% | -0.1825 | 5.3% |

## 3. 数字标记

### 主导
- 本轮无 magnitude_share > 80% 的组件。

### 死组件
- `success_proxy`：active_rate = 0.1%（< 2%）
- `contact_quality`：active_rate = 0.2%（< 2%）

### 淹没
- 无（所有 active_rate > 90% 的组件 magnitude_share 均 ≥ 3%）

### 全覆盖惩罚
- `goal_proximity`：active_rate = 100.0%，signed_share = -53.8%
- `velocity_penalty`：active_rate = 99.8%，signed_share = -32.0%
- `angle_penalty`：active_rate = 100.0%，signed_share = -13.4%

### 量级对比（每步均值 = episode_sum_mean ÷ len）

| 排名 | 组件 | 每步均值 |
|:---:|---:|---:|
| 最大 1 | success_proxy | +0.02096 |
| 最大 2 | contact_quality | +0.01124 |
| 最小 1 | goal_proximity | -2.293 |
| 最小 2 | velocity_penalty | -1.366 |

### 修改变化（相对于上一轮 iter‑02）

| 组件 | Δ episode_sum_mean | Δ active_rate |
|---|---:|---:|
| goal_proximity | +3611.50 | 0.0% |
| velocity_penalty | -69.29 | -0.2% |
| angle_penalty | -29.04 | 0.0% |
| contact_quality | +0.77 | +0.2% |
| success_proxy | -5790.80 | -99.9% |
| thrust_penalty | +47.19 | -94.7% |