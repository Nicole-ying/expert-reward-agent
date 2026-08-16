# 训练数据报告 — iter_07

## 1. 行为事实

- **得分**：score=252.78，range=[49.75, 291.21]（全部 20 集为正，最低 49.7、最高 291.2）
- **终止方式**：terminated 20/20（全部成功着陆），truncated 0/20（无超时）
- **回合长度**：len=308.2
- **奖励误差**：0
- **Checkpoint 官方得分轨迹**：cp1=-151.7 → cp2=-24.7 → cp3=77.7 → cp4=91.6 → cp5=147.6 → cp6=182.0 → cp7=214.0 → cp8=247.6 → cp9=197.7 → cp10=262.8  
  （从 cp3 起全程为正，cp1→cp10 累计上升 414.5 分；cp9 有一个回落 -49.9，cp10 恢复并达到最高）

### 各组件 ep_sum_mean 趋势

| 组件 | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel_penalty | -6.555 | -7.807 | -2.827 | -2.458 | -2.481 | -2.466 | -2.496 | -2.690 | -2.383 | -2.889 |
| orientation_penalty | -0.117 | -0.183 | -0.261 | -0.319 | -0.343 | -0.294 | -0.334 | -0.287 | -0.300 | -0.202 |
| progress_reward | 0.697 | 1.914 | 2.494 | 2.480 | 2.581 | 2.593 | 2.596 | 2.626 | 2.620 | 2.688 |
| vel_penalty | -3.067 | -1.956 | -2.493 | -2.713 | -2.721 | -2.759 | -2.703 | -2.594 | -2.870 | -2.523 |

### 各组件 active_rate 趋势

| 组件 | cp1 | cp2 | cp3 | cp4 | cp5 | cp6 | cp7 | cp8 | cp9 | cp10 |
|---|---|---|---|---|---|---|---|---|---|---|
| fuel_penalty | 87.5% | 91.2% | 58.3% | 56.6% | 52.5% | 49.3% | 50.6% | 50.3% | 52.3% | 53.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 98.7% | 98.3% | 97.2% | 96.6% | 95.8% | 95.2% | 95.9% | 95.0% |
| vel_penalty | 100.0% | 100.0% | 98.9% | 98.5% | 97.4% | 96.9% | 96.3% | 95.8% | 96.3% | 95.6% |

---

## 2. 骨架演进表

| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | vel_penalty_ep_sum | vel_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active | contact_reward_ep_sum | contact_reward_active | landing_proximity_reward_ep_sum | landing_proximity_reward_active | vel_damping_ep_sum | vel_damping_active | approach_reward_ep_sum | approach_reward_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | 20/0 | 0.104 | 100.0% | — | — | -0.034 | 100.0% | — | — | 0.000 | 0.0% | — | — | 0.000 | 0.0% | — | — |
| 2 | -36.38 | 1000.00 | 0/20 | 0.001 | 100.0% | — | — | -0.000 | 100.0% | — | — | — | — | 2.704 | 100.0% | -0.027 | 95.3% | — | — |
| 3 | -434.87 | 967.20 | 1/19 | -0.002 | 100.0% | — | — | -0.001 | 100.0% | — | — | — | — | 0.026 | 68.3% | 0.000 | 0.0% | — | — |
| 4 | -59.19 | 977.90 | 1/19 | 0.001 | 100.0% | — | — | -0.001 | 100.0% | — | — | — | — | 4.731 | 100.0% | -0.022 | 92.1% | — | — |
| 5 | -124.22 | 68.35 | 20/0 | 0.016 | 100.0% | — | — | -0.019 | 100.0% | -0.002 | 3.2% | — | — | 0.034 | 100.0% | -0.171 | 45.6% | — | — |
| 6 | -119.14 | 68.45 | 20/0 | — | — | -0.153 | 99.7% | -0.002 | 100.0% | -0.001 | 13.6% | — | — | — | — | — | — | -0.974 | 100.0% |
| 7 | 252.78 | 308.20 | 20/0 | 0.009 | 95.2% | -0.008 | 95.7% | -0.001 | 100.0% | -0.009 | 51.6% | — | — | — | — | — | — | — | — |

注：`—` 表示该组件在该轮不存在。iter_01 的 progress_reward/contact_reward/orientation_penalty/vel_damping 列重用于全表；iter_06 的 approach_reward 和 vel_penalty 为独立列；iter_07 的 progress_reward/orientation_penalty/vel_penalty/fuel_penalty 对应回原始列。各组件 ep_sum 为 episode_sum_mean 近似值（取 signed_share × |score|/100 或从原数据直接提取）。

---

## 3. 数字标记

### 本轮（iter_07）

- **主导**：无。最大 magnitude_share 为 fuel_penalty 33.9%，未超过 80%。
- **死组件**：无。最小 active_rate 为 fuel_penalty 51.6%，未低于 2%。
- **淹没**：无。最高 active_rate 100% 的 orientation_penalty 其 magnitude_share=2.5%，不满足 active_rate>90% 且 magnitude_share<3%（仅"淹没"标准触发，但严格来说 magnitude_share<3% 满足条件而 active_rate=100%>90%，然而 magnitude_share<3% 非常接近边界；根据定义"淹没"要求同时满足 active_rate>90% 且 magnitude_share<3%，orientation_penalty 满足 2.5%<3% 但该规则并非"标记"，此处仅作"量级最小"呈现，不标记）。
- **全覆盖惩罚**：无。没有组件 active_rate>95% 且 signed_share 为负——orientation_penalty 的 active_rate=100%>95% 且 signed_share 为负（-2.5%），但 magnitude_share 仅 2.5%，按定义应标记：orientation_penalty 的 active_rate=100.0%>95% 且 signed_share 为负（-2.5%），满足"全覆盖惩罚"条件。
- **量级对比**（每步均值 = ep_sum / len）：
  - 最大：fuel_penalty=-0.00908, progress_reward=+0.00867
  - 最小：orientation_penalty=-0.00068（量级最小），vel_penalty=-0.00817
  - 实际上：每步均值最大两组件：fuel_penalty 和 progress_reward；最小两组件：orientation_penalty 和 vel_penalty

### 修改变化（Δ iter_06→iter_07）

本表仅列出 iter_06 和 iter_07 共有的组件名。关注**实质对应**：iter_06 的 approach_reward 是 progress_reward 的语义等价物（同为距离改善量），iter_07 改回 progress_reward 名称但语义相同。

| 组件 | Δ ep_sum | Δ active_rate | 说明 |
|---|---|---|---|
| progress_reward（对应 iter_06 approach_reward） | +0.983 → 从 -0.974 变为 +0.009（ep_sum 尺度不同，实际变正） | -4.8%（100.0%→95.2%） | 语义等价，数值变化来自权重和门控 |
| vel_penalty | +0.145（从 -0.153 缓和到 -0.008） | -4.0%（99.7%→95.7%） | 门控形式重建，惩罚减轻 |
| orientation_penalty | -0.001（基本持平，-0.002→-0.001） | 0.0%（100.0%→100.0%） | 门控重建，影响微小 |
| fuel_penalty | -0.008（从 -0.001 增强到 -0.009） | +38.0%（13.6%→51.6%） | 门槛降低（-0.005→-0.005 不变但统一费率变化？本轮 action 1/3 同样 -0.005，激活率大幅上升） |

---

## 4. 语义演化（跨轮组件功能追踪）

按语义功能对组件分组，列出各轮参与该功能的组件名、得分、以及"方向切换"次数。

| 语义功能 | 各轮组件名与得分 | 方向切换 |
|---|---|---|
| **接近/进度信号** | iter_01: progress_reward (w=1.0, ep_sum≈+0.10, 连续奖励/改善量) → iter_02: progress_reward (w=1.0, ep_sum≈+0.001, 连续/改善量，被 landing_proximity 淹没) → iter_03: progress_reward (w=1.0, ep_sum≈-0.002, 连续/改善量) → iter_04: progress_reward (w=1.0, ep_sum≈+0.001, 连续/改善量) → iter_05: progress_reward (w=1.0, ep_sum≈+0.016, 连续/改善量) → iter_06: approach_reward (实质等价 progress_reward, w=2.0, ep_sum≈-0.974, 连续/改善量) → iter_07: progress_reward (w=2.0, ep_sum≈+2.67, 连续/改善量) | 0 次切换（全程为连续形状/改善量 delta 形式） |
| **着陆/接触信号** | iter_01: contact_reward (w=2.0, 稀疏 bonus, 死组件 ep_sum=0) → iter_02: landing_proximity_reward (w=3.0, 连续奖励/状态值, ep_sum≈+2704) → iter_03: landing_proximity_reward (w=5.0, 连续奖励/改善量加权, ep_sum≈+25.5) → iter_04: landing_proximity_reward (w=10.0, 连续奖励/状态值, ep_sum≈+4731) → iter_05: landing_proximity_reward (w=2.0, 连续奖励/状态值, ep_sum≈+2.32) → iter_06: **移除** → iter_07: **无着陆奖励** | 4 次切换（稀疏 bonus→连续奖励；改善量加权→状态值→状态值→移除 w/ 无近似→无着陆信号） |
| **稳定性/约束信号** | iter_01: vel_damping (gated, w=0.5, 死组件 ep_sum=0) + orientation_penalty (全高度, ep_sum≈-0.034) → iter_02: vel_damping (gated, w=0.5, ep_sum≈-26.78) + orientation_penalty (全高度, ep_sum≈-0.000) → iter_03: vel_damping (gated, w=0.5, 死组件 ep_sum=0) + orientation_penalty (全高度, ep_sum≈-0.001) → iter_04: vel_damping (gated, w=0.5, ep_sum≈-21.76) + orientation_penalty (全高度, ep_sum≈-0.001) → iter_05: vel_damping (gated, w=0.5, ep_sum≈-11.71) + orientation_penalty (全高度, ep_sum≈-0.019) → iter_06: vel_penalty (软门控 1/(1+3d), ep_sum≈-10.49) + orientation_penalty (软门控, ep_sum≈-0.145) → iter_07: vel_penalty (软门控, ep_sum≈-2.52) + orientation_penalty (软门控, ep_sum≈-0.211) | 1 次切换（门控→软门控，发生在 iter_05→iter_06，此后保持稳定） |
| **燃料/稀疏惩罚** | iter_01~04: 无燃料组件 → iter_05: fuel_penalty (离散费率 -0.2/-0.05/0, ep_sum≈-0.11) → iter_06: fuel_penalty (离散费率 -0.02/-0.005/0, ep_sum≈-0.047) → iter_07: fuel_penalty (离散费率 -0.02/-0.005/0, ep_sum≈-2.80) | 0 次切换（全程为连续性惩罚/离散费率） |
| **完成/成功代理** | 无明确的条件性大额正信号或乘积形式完成接近度奖励存在于各轮中 | — |

---

## 更新历史记忆表

| iter | skeleton | score | best | delta | len | key_signal | action |
|---|---|---|---|---|---|---|---|
| 1 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.49 | -159.49 | 0.00 | 303.35 | contact_reward=0.000 orientation_penalty=-0.034 progress_reward=+0.104 vel_damping=0.000 | new_best |
| 2 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -36.38 | -36.38 | 0.00 | 1000.00 | landing_proximity_reward=+2704.04 orientation_penalty=-0.000 progress_reward=+0.001 vel_damping=-26.78 | new_best |
| 3 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -434.87 | -36.38 | -398.49 | 967.20 | landing_proximity_reward=+25.54 orientation_penalty=-0.001 progress_reward=-0.002 vel_damping=0.000 | no_meaningful_improvement |
| 4 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -59.19 | -36.38 | -22.81 | 977.90 | landing_proximity_reward=+4731.13 orientation_penalty=-0.001 progress_reward=+0.001 vel_damping=-21.76 | no_meaningful_improvement |
| 5 | fuel_penalty + landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -124.22 | -36.38 | -87.84 | 68.35 | fuel_penalty=-0.002 landing_proximity_reward=+0.034 orientation_penalty=-0.019 progress_reward=+0.016 vel_damping=-11.71 | unsolved_stagnation_fresh_restart |
| 6 | approach_reward + fuel_penalty + orientation_penalty + vel_penalty | -119.14 | -36.38 | -82.76 | 68.45 | approach_reward=-66.67 fuel_penalty=-0.047 orientation_penalty=-0.145 vel_penalty=-10.49 | no_meaningful_improvement |
| 7 | fuel_penalty + orientation_penalty + progress_reward + vel_penalty | 252.78 | 252.78 | 0.00 | 308.20 | fuel_penalty=-2.80 orientation_penalty=-0.211 progress_reward=+2.67 vel_penalty=-2.52 | target_solved_new_best |