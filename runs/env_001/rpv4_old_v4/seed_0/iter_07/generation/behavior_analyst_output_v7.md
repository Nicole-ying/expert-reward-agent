# 训练数据报告 — iter 06

## 1. 行为事实

- **得分**：score=-119.14，range=[-143.63, -90.80]
- **终止方式**：terminated=20/20（全部提前失败），truncated=0/20（无时间截断）
- **回合长度**：len=68.45
- **Checkpoint 轨迹**：
  - 得分趋势：cp10%=-116.6 → cp100%=-121.5（基本稳定在 -120 附近，无显著改善）
  - approach_reward ep_sum：始终在 -67 左右，全程扁平
  - vel_penalty ep_sum：始终在 -10.4 左右，全程扁平
  - orientation_penalty ep_sum：全程接近 0（-0.4 到 -0.1）
  - fuel_penalty ep_sum：全程接近 0（-0.000 到 -0.121），active_rate 从 cp10% 的 0.1% 逐渐下降到 cp100% 的 12.7%

## 2. 骨架演进表

| iter | score | len | term | progress/approach_ep_sum | progress/approach_active | orientation_penalty_ep_sum | orientation_penalty_active | contact/landing_ep_sum | contact/landing_active | vel_damping/vel_penalty_ep_sum | vel_damping/vel_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | T20 | 0.104 | 100% | -0.034 | 100% | 0.000 | 0% | 0.000 | 0% | — | — |
| 2 | -36.38 | 1000.00 | S20 | 1.037 | 100% | -0.302 | 100% | 2704.041 | 100% | -26.778 | 95.3% | — | — |
| 3 | -434.87 | 967.20 | T1/S19 | -1.975 | 100% | -0.874 | 100% | 25.542 | 68.3% | 0.000 | 0% | — | — |
| 4 | -59.19 | 977.90 | T1/S19 | 0.837 | 100% | -0.558 | 100% | 4731.133 | 100% | -21.756 | 92.1% | — | — |
| 5 | -124.22 | 68.35 | T20 | 1.114 | 100% | -1.330 | 100% | 2.316 | 100% | -11.708 | 45.6% | -0.110 | 3.2% |
| 6 | -119.14 | 68.45 | T20 | -66.672 | 100% | -0.145 | 100% | — | — | -10.486 | 99.7% | -0.047 | 13.6% |

**表注**：
- iter 1–5 的进度组件为 `progress_reward`（delta型），iter 6 为 `approach_reward`（状态值型）
- iter 1 的着陆组件为 `contact_reward`（乘积连续型），iter 2–5 为 `landing_proximity_reward`（状态Gaussian型），iter 6 无此组件
- iter 1 的速度组件为 `vel_damping`（门控惩罚型，死组件），iter 2–5 为 `vel_damping`（门控惩罚型），iter 6 为 `vel_penalty`（门控惩罚型，公式形态不同）
- T20=terminated 20/20，S20=truncated 20/20

## 3. 数字标记

- **主导**：`approach_reward` — magnitude_share=86.2%（>80%）
- **全覆盖惩罚**：`approach_reward` — active_rate=100%（>95%）且 signed_share=-86.2%（为负）
- **接近淹没**：`orientation_penalty` — active_rate=100%（>90%）但 magnitude_share=0.2%（<3%），未达到严格淹没阈值但极低
- **死组件**：`fuel_penalty` — active_rate=13.6%（未达到<2%的严格死组件标准，但活性较低）
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大：`approach_reward` = -0.974 / step；`vel_penalty` = -0.153 / step
  - 最小：`orientation_penalty` = -0.002 / step；`fuel_penalty` = -0.001 / step
- **修改变化（iter 5 → iter 6）**：
  - `progress_reward` → `approach_reward`：ep_sum 从 +1.114 → -66.672（Δ=-67.786），active_rate 保持 100%
  - `orientation_penalty`：ep_sum 从 -1.330 → -0.145（Δ=+1.185），active_rate 保持 100%
  - `vel_damping` → `vel_penalty`：ep_sum 从 -11.708 → -10.486（Δ=+1.222），active_rate 从 45.6% → 99.7%（Δ=+54.1pp）
  - `fuel_penalty`：ep_sum 从 -0.110 → -0.047（Δ=+0.063），active_rate 从 3.2% → 13.6%（Δ=+10.4pp）
  - `landing_proximity_reward`：已移除（iter 5 ep_sum=+2.316, active_rate=100%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_progress_reward(改善量) → v2_progress_reward(改善量) → v3_progress_reward(改善量) → v4_progress_reward(改善量) → v5_progress_reward(改善量) → v6_approach_reward(状态值) | -159 → -36 → -435 → -59 → -124 → -119 | 改善量→状态值：1次切换（v5→v6） |
| 着陆/接触信号 | v1_contact_reward(死) → v2_landing_proximity(全主导99%) → v3_landing_proximity(活性68%) → v4_landing_proximity(全主导99.5%) → v5_landing_proximity(小额13.9%) → v6_无 | -159 → -36 → -435 → -59 → -124 → -119 | 组件存活→移除：1次（v5→v6）；形态：乘积连续→状态Gaussian（v1→v2），1次；宽度：宽sigma(2.0)→窄(0.7)→宽(0.7保持)（v2→v4），1次；加权：状态值→进度加权→状态值（v2→v4），2次切换。合计在该功能上发生至少5次设计振荡 |
| 稳定性/约束信号 | v1_vel_damping(死) → v2_vel_damping(门控1.0%) → v3_vel_damping(死) → v4_vel_damping(门控0.5%) → v5_vel_damping(门控70.5% dominate) → v6_vel_penalty(门控13.6%) | -159 → -36 → -435 → -59 → -124 → -119 | 活性：死→活→死→活→主导→次要，5次切换；公式：abs求和→平方求和（v5→v6），1次；惩罚对象：vx+vy → vx²+vy²+0.1*angvel²（v5→v6），1次 |
| 完成/成功代理 | v5_fuel_penalty(稀疏3.2%) → v6_fuel_penalty(稀疏13.6%) | -124 → -119 | 无方向切换，权重降低（-0.2→-0.02主发动机） |
| 燃料效率（交叉功能） | v5_fuel_penalty(稀疏惩罚) → v6_fuel_penalty(稀疏惩罚) | -124 → -119 | 权重从 -0.2/-0.05/-0.0 → -0.02/-0.005/-0.0，降低了10倍，仍在稀疏使用 |