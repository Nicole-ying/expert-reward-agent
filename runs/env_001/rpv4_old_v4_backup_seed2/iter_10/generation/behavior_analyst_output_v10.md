# 训练数据报告 — 本轮 (iter=9)

---

## 1. 行为事实

- **得分**：score=186.54，range=[14.38, 288.21]
- **终止方式**：terminated=20/20（成功），truncated=0/20（时间截断）
- **回合长度**：len=260.90
- **Checkpoint 轨迹**（官方得分）：cp10%=-106.3 → cp20%=55.0 → cp30%=22.1 → cp40%=-1.3 → cp50%=-4.8 → cp60%=-23.9 → cp70%=-22.9 → cp80%=39.4 → cp90%=125.8 → cp100%=188.1（总体上升，cp30-70 经历性能低谷）

---

## 2. 骨架演进表

（注：iter 1-5 骨架结构与 iter 6-9 不同，分别列出）

| iter | score | len | term | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | landing_ep_sum | landing_active | velocity_guidance_ep_sum | velocity_guidance_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -203.53 | 185.80 | 20T/0Tr | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | -37.03 | 962.55 | 4T/16Tr | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | — | — |
| 3 | -222.85 | 209.60 | 20T/0Tr | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | — | — |
| 4 | -295.33 | 138.90 | 20T/0Tr | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | — | — | -1.596 | 100.0% |
| 5 | -197.46 | 185.50 | 20T/0Tr | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | 0.000 | 0.0% | — | — |

| iter | score | len | term | distance_progress_ep_sum | distance_progress_active | stability_ep_sum | stability_active | descent_bonus_ep_sum | descent_bonus_active | landing_completion_ep_sum | landing_completion_active | fuel_penalty_ep_sum | fuel_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | -13.57 | 1000.00 | 0T/20Tr | 6.331 | 100.0% | -1.416 | 100.0% | — | — | 0.000 | 0.0% | -9.889 | 98.9% |
| 7 | 146.32 | 673.35 | 14T/6Tr | 5.730 | 99.5% | -3.284 | 100.0% | 3.276 | 62.8% | 47.039 | 19.0% | -4.572 | 67.9% |
| 8 | 199.42 | 477.65 | 17T/3Tr | 6.304 | 98.5% | -1.871 | 100.0% | 6.377 | 69.7% | 70.983 | 32.3% | -3.051 | 63.9% |
| **9** | **186.54** | **260.90** | **20T/0Tr** | **6.257** | **96.8%** | **-1.923** | **100.0%** | **6.856** | **84.5%** | **19.172** | **19.4%** | **-1.384** | **53.0%** |

*T=terminated, Tr=truncated*

---

## 3. 数字标记

- **主导**：无。landing_completion magnitude_share=53.2%，未超过 80%。
- **死组件**：无。所有组件 active_rate ≥ 19.4%（最低为 landing_completion=19.4%）。
- **淹没**：无。stability active_rate=100% 但 magnitude_share=5.3%，未低于 3%；distance_progress active_rate=96.8%，magnitude_share=18.6%，未低于 3%。
- **全覆盖惩罚**：stability signed_share 为负（-5.3%），active_rate=100%。符合条件（active_rate>95% 且 signed_share 为负）。
- **量级对比（每步均值 = ep_sum ÷ len）**：
  - 最大两个：landing_completion=0.0735，descent_bonus=0.0263
  - 最小两个：fuel_penalty=-0.0053，stability=-0.0074
- **修改变化（iter 8 → iter 9 Δ）**：
  - distance_progress: ep_sum Δ=-0.047, active_rate Δ=-1.7%
  - stability: ep_sum Δ=-0.052, active_rate Δ=0.0%
  - descent_bonus: ep_sum Δ=+0.479, active_rate Δ=+14.8%
  - landing_completion: ep_sum Δ=-51.811, active_rate Δ=-12.9%
  - fuel_penalty: ep_sum Δ=+1.667, active_rate Δ=-10.9%

---

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向/设计切换 |
|---|---|---|---|---|
| 距离/进度引导 | iter1: progress → iter2: progress → iter3: progress → iter4: progress → iter5: progress → iter6: distance_progress → iter7: distance_progress → iter8: distance_progress → iter9: distance_progress | -0.200 → 0.665 → -0.299 → -0.026 → -0.183 → 6.331 → 5.730 → 6.304 → 6.257 | 100% → 100% → 100% → 100% → 100% → 100% → 99.5% → 98.5% → 96.8% | iter6 重构：原始距离差 → 5×缩放；iter4 追加 velocity_guidance（速度方向引导，iter5 移除）|
| 垂直下降激励 | iter6: descent_bonus（无）→ iter7: descent_bonus → iter8: descent_bonus → iter9: descent_bonus | — → 3.276 → 6.377 → 6.856 | — → 62.8% → 69.7% → 84.5% | iter7 引入（系数 6.0, x_gate /0.5）；iter9 x_gate 从 /0.5 扩至 /0.8 |
| 速度方向引导 | iter4: velocity_guidance → iter5: 移除 | -1.596 → — | 100% → — | 1 次引入→移除 |

### 着陆/接触信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 着陆/接触奖励 | iter1: landing → iter2: landing → iter3: landing → iter4: 移除 → iter5: landing → iter6: landing_completion → iter7: landing_completion → iter8: landing_completion → iter9: landing_completion | 0.000 → 697.436 → 0.000 → — → 0.000 → 0.000 → 47.039 → 70.983 → 19.172 | 0% → 99.6% → 0% → — → 0% → 0% → 19.0% → 32.3% → 19.4% | 稀疏bonus↔连续奖励：4次切换（iter1 接触门控相乘 → iter2 连续 proximity-gated sum-of-factors → iter3 接触门控 quality-scaled → iter5 proximity-gated contact-multiplier → iter6 硬条件 sparse bonus）；iter7-9 稳定为硬条件 sparse bonus |

### 稳定性/约束信号

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 速度约束 | iter1: speed_penalty → iter2: speed_penalty → iter3: speed_penalty → iter4: speed_penalty → iter5: speed_penalty → iter6: stability → iter7: stability → iter8: stability → iter9: stability | -0.669 → -42.130 → -0.735 → -2.097 → -0.717 → -1.416 → -3.284 → -1.871 → -1.923 | 15.5% → 100% → 14.8% → 63.2% → 15.8% → 100% → 100% → 100% → 100% | iter6 重构：线性 closeness-gated speed 惩罚 → exponential-gated 平方惩罚（含 angle, ang_vel）；iter1-5 为距离门控线性惩罚，iter6-9 为 exp(-2d) 门控平方惩罚 |
| 姿态约束 | iter1: angular_vel_penalty → iter2: angular_vel_penalty → iter3: angular_vel_penalty → iter4: angular_vel_penalty → iter5: angular_vel_penalty → iter6: (合并入 stability) | -0.434 → -2.457 → -0.368 → -1.266 → -0.343 → 合并 | 100% → 100% → 100% → 100% → 100% → 合并 | iter6 将角速度惩罚合并入 stability 组件（angle + ang_vel 平方） |
| 燃料约束 | iter6: fuel_penalty → iter7: fuel_penalty → iter8: fuel_penalty → iter9: fuel_penalty | -9.889 → -4.572 → -3.051 → -1.384 | 98.9% → 67.9% → 63.9% → 53.0% | 无切换，稳定为恒定 -0.01 per action |

### 完成/成功代理

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮 active_rate | 方向切换 |
|---|---|---|---|---|
| 成功代理 | iter1-5: landing（同上）→ iter6: landing_completion → iter7: landing_completion → iter8: landing_completion → iter9: landing_completion | 见着陆/接触信号行 | 见着陆/接触信号行 | iter1-5 landing 在 0 和非零之间振荡；iter6-9 landing_completion 为硬条件乘积 sparse bonus |