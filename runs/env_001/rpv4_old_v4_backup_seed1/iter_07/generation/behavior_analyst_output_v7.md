## 1. 行为事实

- 得分：score=-81.80，range=[-149.22, 84.98]
- 终止方式：terminated 0/20（无成功着陆），truncated 20/20（全部时间上限截断）
- 回合长度：len=1000.00（全部跑满1000步）
- 最高分84.98出现在某评估集，远低于历史最佳265.83（iter_02）
- Checkpoint 轨迹：cp1=-172.8 → cp2=71.4 → cp3=-39.0 → cp4=34.8 → cp5=62.3 → cp6=61.1 → cp7=138.5 → cp8=56.2 → cp9=46.2 → cp10=-67.1
  - 最高官方得分出现在cp7（138.5），随后持续下降至cp10的-67.1
  - cp10得分（-67.1）与cp2得分（71.4）相差138.5，说明训练后期性能大幅退化
  - 与iter_05对比（score=-33.04）：得分下降-48.8，回合长度持平（均为1000）

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | upright_penalty_ep_sum | upright_penalty_active | contact_bonus_ep_sum | contact_bonus_active | landing_approach_ep_sum | landing_approach_active |
|------|-------|-----|------|----------------------|----------------------|------------------------|------------------------|-----------------------|-----------------------|---------------------|---------------------|------------------------|------------------------|
| 1 | -76.56 | 975.15 | 1/20 | 0.635 | 100.0% | -12.561 | 100.0% | -3.714 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | 63.72 | 668.05 | 17/20 | 1.835 | 98.8% | -9.517 | 99.0% | -7.301 | 100.0% | 1075.630 | 5.2% | — | — |
| 3 | 21.45 | 864.55 | 9/20 | 1.769 | 99.9% | -9.103 | 99.9% | -11.452 | 100.0% | 1011.468 | 8.2% | — | — |
| 4 | -10.11 | 1000.00 | 0/20 | 2.922 | 100.0% | -7.837 | 100.0% | -5.113 | 100.0% | 0.000 | 0.0% | — | — |
| 5 | -33.04 | 1000.00 | 0/20 | 4.286 | 100.0% | -10.122 | 100.0% | -3.598 | 100.0% | 0.000 | 0.0% | — | — |
| 6 | -81.80 | 1000.00 | 0/20 | 1.775 | 100.0% | -2.385 | 100.0% | -3.075 | 100.0% | — | — | 0.974 | 98.3% |

## 3. 数字标记

- **主导**：无。最大magnitude_share为upright_penalty的29.4%，远低于80%阈值。

- **死组件**：无。所有组件active_rate均≥98.3%。

- **淹没**：无。无组件同时满足active_rate>90%且magnitude_share<3%。

- **全覆盖惩罚**：无。无组件满足active_rate>95%且signed_share为负——upright_penalty虽为负且active_rate=100%，但magnitude_share=29.4%，不是完全主导；velocity_penalty虽为负且active_rate=100%，但magnitude_share=22.8%，同样不构成全覆盖。

- **量级对比**：
  - 每步均值（ep_sum ÷ len=1000）：
    - goal_proximity: 0.00177
    - velocity_penalty: -0.00239
    - upright_penalty: -0.00308
    - landing_approach: 0.00097
  - 最大两个：upright_penalty (-0.00308，绝对值最大)，velocity_penalty (-0.00239)
  - 最小两个：landing_approach (0.00097)，goal_proximity (0.00177)

- **修改变化**（iter_05 → iter_06）：
  - goal_proximity: ep_sum 4.286→1.775（Δ=-2.511），active_rate 100.0%→100.0%（Δ=0.0pp）
  - velocity_penalty: ep_sum -10.122→-2.385（Δ=+7.737），active_rate 100.0%→100.0%（Δ=0.0pp）
  - upright_penalty: ep_sum -3.598→-3.075（Δ=+0.523），active_rate 100.0%→100.0%（Δ=0.0pp）
  - contact_bonus→landing_approach: ep_sum 0.000→0.974（新组件），active_rate 0.0%→98.3%（Δ=+98.3pp）

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_goal_proximity→v2_goal_proximity→v3_goal_proximity→v4_goal_proximity→v5_goal_proximity→v6_goal_proximity | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | delta距离(iter_01)→delta距离×2(iter_02)→逆势能(iter_04)→delta距离×4(iter_05-06): 1次切换 |
| 着陆/接触信号 | v1_contact_bonus(死)→v2_contact_bonus(5.2%)→v3_contact_bonus(8.2%)→v4_contact_bonus(死)→v5_contact_bonus(死)→v6_landing_approach | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 稀疏bonus(iter_01-05)↔连续势能(iter_06): 1次切换 |
| 稳定性/约束（速度） | v1_velocity_penalty→v2_velocity_penalty→v3_velocity_penalty→v4_velocity_penalty→v5_velocity_penalty→v6_velocity_penalty | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 全高度(iter_01)→门控(iter_02-06): 1次切换 |
| 稳定性/约束（姿态） | v1_upright_penalty→v2_upright_penalty→v3_upright_penalty→v4_upright_penalty→v5_upright_penalty→v6_upright_penalty | -76.56→63.72→21.45→-10.11→-33.04→-81.80 | 无切换(持续active) |