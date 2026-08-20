# 训练数据记者报告 — iter_04

## 1. 行为事实
- **得分**：score = -295.33，range = [-343.97, -239.83]
- **终止方式**：terminated = 20/20（全部成功/失败终止），truncated = 0/20（无超时截断）
- **回合长度**：len = 138.9 步
- **Checkpoint 轨迹**（官方得分）：
  cp10% = -574.8 → cp20% = -653.7 → cp30% = -605.5 → cp40% = -571.7 → cp50% = -514.4 → cp60% = -441.3 → cp70% = -395.5 → cp80% = -334.2 → cp90% = -325.3 → cp100% = -289.6
  整体趋势：得分从 -574.8 连续提升至 -289.6（负值减小）。

## 2. 骨架演进表

| iter | score | len | term | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | landing_ep_sum | landing_active | velocity_guidance_ep_sum | velocity_guidance_active |
|------|-------|------|------|------------------|------------------|-----------------------|-----------------------|----------------------------|----------------------------|----------------|----------------|---------------------------|---------------------------|
| 1 | -203.53 | 185.80 | 20/20,0 | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | – | – |
| 2 | -37.03 | 962.55 | 4/20,16/20 | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | – | – |
| 3 | -222.85 | 209.60 | 20/20,0 | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | – | – |
| 4 | -295.33 | 138.90 | 20/20,0 | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | – | – | -1.596 | 100.0% |

## 3. 数字标记

- **主导**：无组件 magnitude_share > 80%。
- **死组件**：无组件 active_rate < 2%。
- **淹没**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。
- **全覆盖惩罚**（active_rate > 95% 且 signed_share 为负）：
  - `velocity_guidance`：active=100.0%，signed_share=-18.9%
  - `angular_vel_penalty`：active=100.0%，signed_share=-15.0%
  - `progress`：active=100.0%，signed_share=-0.3%
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大两个（负值中数值最大，即最不负面）：
    - `progress` ≈ -0.00019
    - `angular_vel_penalty` ≈ -0.00911
  - 最小两个（负值中数值最小，即最负面）：
    - `speed_penalty` ≈ -0.01509
    - `velocity_guidance` ≈ -0.01149
- **修改变化**（相比上一轮 iter_03）：
  - `progress`：Δ ep_sum = +0.273（从 -0.299 → -0.026），Δ active = +0 pp（100% → 100%）
  - `speed_penalty`：Δ ep_sum = -1.361（从 -0.735 → -2.097），Δ active = +48.4 pp（14.8% → 63.2%）
  - `angular_vel_penalty`：Δ ep_sum = -0.898（从 -0.368 → -1.266），Δ active = +0 pp（100% → 100%）
  - `landing`：移除（原 ep_sum=0.000，active=0.0%）
  - `velocity_guidance`：新增（ep_sum=-1.596，active=100.0%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum） | 方向切换 |
|---------|----------|-----------------|---------|
| 接近/进度信号 | v1_progress, v2_progress, v3_progress, v4_progress + velocity_guidance | -0.200, 0.665, -0.299, progress=-0.026 / velocity_guidance=-1.596 | 仅增加方向引导组件，未在功能设计上切换 |
| 着陆/接触信号 | v1_landing, v2_landing, v3_landing, v4_(无) | 0.000, 697.436, 0.000, – | 稀疏bonus↔连续奖励：2次切换（v1乘积门控→v2 sum-of-factors连续；v2连续→v3 contact-gated稀疏） |
| 稳定性约束（速度惩罚） | v1_speed_penalty, v2_speed_penalty, v3_speed_penalty, v4_speed_penalty | -0.669, -42.130, -0.735, -2.097 | 无切换（始终为靠近减速惩罚） |
| 稳定性约束（角速度惩罚） | v1_angular_vel_penalty, v2_angular_vel_penalty, v3_angular_vel_penalty, v4_angular_vel_penalty | -0.434, -2.457, -0.368, -1.266 | 无切换（始终为恒定角速度惩罚） |