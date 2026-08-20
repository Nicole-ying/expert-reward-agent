# 数据报告

## 1. 行为事实
- 得分: score = -12.569633, range = [-43.381619, 24.541211]
- 终止方式: terminated = 0/20 (成功率 0%), truncated = 20/20 (时间上限率 100%)
- 回合长度: len = 1000.000000
- Checkpoint 轨迹: 无数据

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | contact_quality_ep_sum | contact_quality_active | success_proxy_ep_sum | success_proxy_active | soft_success_ep_sum | soft_success_active | thrust_penalty_ep_sum | thrust_penalty_active | goal_distance_progress_ep_sum | goal_distance_progress_active | soft_landing_ep_sum | soft_landing_active | orientation_penalty_ep_sum | orientation_penalty_active | landing_contact_success_ep_sum | landing_contact_success_active |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -121.00 | 68.35 | 20/20,0/20 | -148.505275 | 100.0% | -93.225748 | 99.9% | -43.986293 | 100.0% | 0.766166 | 0.2% | 1.432261 | 0.1% | N/A | N/A | -0.215000 | 6.3% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 2 | -183.32 | 947.35 | 7/20,13/20 | -3768.392629 | 100.0% | -24.149682 | 100.0% | -9.949903 | 100.0% | 0.000000 | 0.0% | 5792.231178 | 100.0% | N/A | N/A | -47.367500 | 100.0% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 3 | -120.25 | 68.40 | 20/20,0/20 | -156.891027 | 100.0% | -93.439696 | 99.8% | -38.994126 | 100.0% | 0.768669 | 0.2% | 1.434087 | 0.1% | N/A | N/A | -0.182500 | 5.3% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 4 | -1471.59 | 435.40 | 20/20,0/20 | 163.929425 | 100.0% | -376.896537 | 100.0% | -3.785140 | 100.0% | 0.000000 | 0.0% | N/A | N/A | 12716.349643 | 96.2% | -21.770000 | 100.0% | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | -63.97 | 963.75 | 1/20,19/20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 3.779845 | 100.0% | -4.930737 | 100.0% | -1.218466 | 100.0% | 0.000000 | 0.0% |
| 6 | -37.58 | 947.65 | 3/20,17/20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 4.980184 | 100.0% | -2.922441 | 100.0% | -2.135488 | 100.0% | 2.312616 | 0.1% |
| 7 | 156.75 | 577.95 | 16/20,4/20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 5.831261 | 99.1% | -5.571498 | 99.2% | -2.558454 | 100.0% | 110.223827 | 17.8% |
| 8 | -18.99 | 1000.00 | 0/20,20/20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 6.119317 | 100.0% | -6.286990 | 100.0% | -1.474458 | 100.0% | 0.000000 | 0.0% |
| 9 | -12.57 | 1000.00 | 0/20,20/20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 6.281804 | 100.0% | -1.594844 | 100.0% | -0.643557 | 100.0% | 0.000000 | 0.0% |

*注: term 列为 `terminated/20, truncated/20`*

## 3. 数字标记

### 本轮 (iter 9) 标记
- **死组件**: `landing_contact_success` (active_rate = 0.0% < 2%)
- **全覆盖惩罚**: `soft_landing` (active_rate = 100.0% > 95%, signed_share = -6.7%) 和 `orientation_penalty` (active_rate = 100.0% > 95%, signed_share = -16.5%)
- **量级对比** (每步均值 = ep_sum / len):
  - 最大两步均值: `goal_distance_progress` (0.00628), `landing_contact_success` (0.0)
  - 最小两步均值: `soft_landing` (-0.00159), `orientation_penalty` (-0.00064)

### 与上一轮 (iter 8) 对比的 Δ
| 组件 | Δ ep_sum_mean | Δ active_rate |
|---|---|---|
| goal_distance_progress | +0.162487 | 0 pp |
| soft_landing | +4.692146 | 0 pp |
| orientation_penalty | +0.830901 | 0 pp |
| landing_contact_success | 0.0 | 0 pp |