# 数据报告

## 1. 行为事实

- **得分**：score=-5.37，range=[-37.70, 118.13]
- **终止方式**：terminated 2/20（成功着陆），truncated 18/20（超时）。仅10%的episode在时限内完成着陆。
- **回合长度**：len=964.10步（最大1000步）
- **Checkpoint轨迹**：官方得分从cp10%=-116.2 → cp100%=-13.5，中间剧烈波动（cp30%=118.6峰值，cp60%=-67.4低谷），最终得分未稳定到正值区域。

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | upright_penalty_ep_sum | upright_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active | landing_approach_ep_sum | landing_approach_active |
|------|-------|-----|------|---------------------|---------------------|----------------------|----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 1 | -76.56 | 975.15 | 1/20 | 0.000 | 0.0% | 0.635 | 100.0% | -3.714 | 100.0% | -12.561 | 100.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1075.630 | 5.2% | 1.835 | 98.8% | -7.301 | 100.0% | -9.517 | 99.0% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1011.468 | 8.2% | 1.769 | 99.9% | -11.452 | 100.0% | -9.103 | 99.9% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 0.000 | 0.0% | 2.922 | 100.0% | -5.113 | 100.0% | -7.837 | 100.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 0.000 | 0.0% | 4.286 | 100.0% | -3.598 | 100.0% | -10.122 | 100.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | — | — | 1.775 | 100.0% | -3.075 | 100.0% | -2.385 | 100.0% | 0.974 | 98.3% |
| 7 | -5.37 | 964.10 | 2/20 | 95.904 | 0.7% | 4.754 | 100.0% | -5.455 | 100.0% | -4.621 | 100.0% | — | — |

## 3. 数字标记

- **主导**：contact_bonus magnitude_share=86.1%（>80%阈值）。尽管active_rate仅0.7%，其稀疏大额奖励在总奖励中占主导地位。
- **死组件**：contact_bonus active_rate=0.7%（<2%阈值）。在几乎所有时间步中不触发，仅在着陆瞬间生效。
- **淹没**：
  - goal_proximity active_rate=100% 但 magnitude_share=4.8%（<3%阈值失效，但接近）。实际上goal_proximity magnitude_share为4.8%，未触发<3%阈值——更正：不标记为淹没。
  - 重新检查：magnitude_share < 3%的组件：无。goal_proximity=4.8%, upright_penalty=4.9%, velocity_penalty=4.1%，均超过3%。
  - **修正**：本轮无淹没组件。
- **全覆盖惩罚**：
  - upright_penalty active_rate=100% 且 signed_share=-4.9%（负值），满足条件。
  - velocity_penalty active_rate=100% 且 signed_share=-4.1%（负值），满足条件。
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大两个：contact_bonus=0.0995，goal_proximity=0.00493
  - 最小两个：velocity_penalty=-0.00479（绝对值排序：upright_penalty=-0.00566, velocity_penalty=-0.00479为两个最小量级）
- **修改变化（iter6→iter7）**：
  - contact_bonus：ep_sum 从无→95.904（新增），active_rate 从无→0.7%
  - goal_proximity：ep_sum 1.775→4.754（Δ+2.979），active_rate 100.0%→100.0%（Δ=0）
  - upright_penalty：ep_sum -3.075→-5.455（Δ-2.380），active_rate 100.0%→100.0%（Δ=0）
  - velocity_penalty：ep_sum -2.385→-4.621（Δ-2.236），active_rate 100.0%→100.0%（Δ=0）
  - landing_approach：被移除（iter6 ep_sum=0.974, active_rate=98.3%）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | v1_contact_bonus(死) → v2_contact_bonus(主导) → v3_contact_bonus(主导,速度门控) → v4_contact_bonus(死) → v5_contact_bonus(死) → v6_landing_approach(稀疏bonus→连续奖励) → v7_contact_bonus(主导) | v1:-76.6 → v2:63.7 → v3:21.5 → v4:-10.1 → v5:-33.0 → v6:-81.8 → v7:-5.4 | 稀疏bonus↔连续奖励：2次切换(v5→v6, v6→v7)；速度门控添加→移除：1次切换(v2→v3添加, v3→v4移除) |
| 接近/进度信号 | v1_goal_proximity(线性,权重1) → v2_goal_proximity(线性,权重2) → v3_goal_proximity(线性,权重2) → v4_goal_proximity(逆距离势能,权重3) → v5_goal_proximity(线性,权重4) → v6_goal_proximity(线性,权重4) → v7_goal_proximity(线性,权重4) | v1:-76.6 → v2:63.7 → v3:21.5 → v4:-10.1 → v5:-33.0 → v6:-81.8 → v7:-5.4 | 线性↔逆距离势能：1次切换(v3→v4, v4→v5)；权重从1→2→2→3→4→4→4：单调递增 |
| 稳定性/约束信号 | v1_velocity_penalty(-0.5,无门控) → v2_velocity_penalty(-0.5,高度门控) → v3_velocity_penalty(-0.5,高度门控) → v4_velocity_penalty(-0.5,高度门控) → v5_velocity_penalty(-0.5,高度门控) → v6_velocity_penalty(-0.3,高度门控) → v7_velocity_penalty(-0.3,高度门控) | v1:-76.6 → v2:63.7 → v3:21.5 → v4:-10.1 → v5:-33.0 → v6:-81.8 → v7:-5.4 | 无门控→高度门控：1次切换(v1→v2)；权重量级：1次调整(v5→v6: -0.5→-0.3) |
| 稳定性/约束信号 | v1_upright_penalty(-1.0) → v2_upright_penalty(-1.0) → v3_upright_penalty(-1.0) → v4_upright_penalty(-1.0) → v5_upright_penalty(-1.0) → v6_upright_penalty(-1.0) → v7_upright_penalty(-1.0) | v1:-76.6 → v2:63.7 → v3:21.5 → v4:-10.1 → v5:-33.0 → v6:-81.8 → v7:-5.4 | 无切换：全轮次保持一致形态和权重 |