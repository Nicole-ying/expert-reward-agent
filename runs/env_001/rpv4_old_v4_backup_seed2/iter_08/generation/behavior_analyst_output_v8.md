# 数据报告 — iter_07

## 1. 行为事实
- **得分**：score=146.32，range=[0.90, 252.42]
- **终止方式**：terminated 14/20（着陆/坠毁），truncated 6/20（达到时间上限）
- **回合长度**：len=673.35（截断集均约 1000，终止集均约 533）
- **Checkpoint 轨迹**：官方得分 cp10%=-99.0 → cp20%=-36.6 → cp30%=2.6 → cp40%=13.4 → cp50%=2.2 → cp60%=-14.4 → cp70%=30.9 → cp80%=148.8 → cp90%=105.3 → cp100%=179.9（先升后降再升，后期突破正值）
- **组件组成**（ep_sum_mean）：landing_completion=47.04（70.5% signed share），distance_progress=5.73（8.6%），descent_bonus=3.28（4.9%），stability=-3.28（-4.9%），fuel_penalty=-4.57（-6.8%）
- **活跃率**：distance_progress 99.5%，stability 100.0%，fuel_penalty 67.9%，descent_bonus 62.8%，landing_completion 19.0%
- **步均贡献**（ep_sum ÷ len）：landing_completion=0.0699，distance_progress=0.0085，descent_bonus=0.0049，stability=-0.0049，fuel_penalty=-0.0068

## 2. 骨架演进表

| iter | score | len | term | landing_ep_sum | landing_active | progress_ep_sum | progress_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | velocity_guidance_ep_sum | velocity_guidance_active | stability_ep_sum | stability_active | descent_bonus_ep_sum | descent_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | landing_completion_ep_sum | landing_completion_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -203.53 | 185.8 | 20/0/0 | 0.000 | 0.0% | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 2 | -37.03 | 962.6 | 4/16/0 | 697.436 | 99.6% | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 3 | -222.85 | 209.6 | 20/0/0 | 0.000 | 0.0% | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 4 | -295.33 | 138.9 | 20/0/0 | — | — | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | -1.596 | 100.0% | — | — | — | — | — | — | — | — |
| 5 | -197.46 | 185.5 | 20/0/0 | 0.000 | 0.0% | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | — | — | — | — | — | — | — | — | — | — |
| 6 | -13.57 | 1000.0 | 0/20/0 | — | — | 6.331 | 100.0% | — | — | — | — | — | — | -1.416 | 100.0% | 0.000 | 0.0% | -9.889 | 98.9% | 0.000 | 0.0% |
| 7 | 146.32 | 673.4 | 14/6/0 | — | — | 5.730 | 99.5% | — | — | — | — | — | — | -3.284 | 100.0% | 3.276 | 62.8% | -4.572 | 67.9% | 47.039 | 19.0% |

注：progress 在 iter_1-5 中为"progress"，在 iter_6-7 中为"distance_progress"（相同功能，列合并展示）。landing 在 iter_1-5 中为"landing"，在 iter_6-7 中为"landing_completion"（相同功能，列合并展示）。stability 仅在 iter_6-7 出现。

## 3. 数字标记

- **主导**：landing_completion magnitude_share=70.5%（>80%），在 ep_sum 中占绝对主导
- **死组件**：无（所有组件 active_rate ≥ 19.0%）
- **淹没**：无（无组件同时满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：无（无组件满足 active_rate > 95% 且 signed_share 为负；stability active_rate=100% 但 signed_share=-4.9%，未达<-10% 极端负值标准）
- **量级对比**：
  - 步均最大（ep_sum/len）：landing_completion=0.0699，distance_progress=0.0085
  - 步均最小：stability=-0.0049（绝对值排第三），fuel_penalty=-0.0068（绝对值排第二）
- **修改变化（vs iter_06）**：
  - distance_progress：ep_sum 6.33→5.73（Δ=-0.60），active_rate 100.0%→99.5%（Δ=-0.5%）
  - stability：ep_sum -1.42→-3.28（Δ=-1.87），active_rate 100.0%→100.0%（Δ=0）
  - descent_bonus：ep_sum 0.00→3.28（Δ=+3.28），active_rate 0.0%→62.8%（Δ=+62.8%）
  - fuel_penalty：ep_sum -9.89→-4.57（Δ=+5.32），active_rate 98.9%→67.9%（Δ=-31.0%）
  - landing_completion：ep_sum 0.00→47.04（Δ=+47.04），active_rate 0.0%→19.0%（Δ=+19.0%）

## 4. 语义演化

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_progress → v4_progress → v6_distance_progress | -203.53 → -295.33 → 146.32 | 无切换：始终为状态改善量（delta距离），全高度激活 |
| 着陆/接触信号 | v1_landing(死) → v2_landing(主导) → v3_landing(死) → v5_landing(死) → v6_landing_completion(死) → v7_landing_completion | -203.53 → -37.03 → -222.85 → -197.46 → -13.57 → 146.32 | 稀疏bonus↔连续奖励：5次切换（v1乘积稀疏→v2 proximity连续→v3 contact稀疏→v5 proximity连续→v6 双足乘积稀疏→v7 双足乘积稀疏） |
| 稳定性/约束信号 | v1_speed_penalty + v1_angular_vel_penalty → v6_stability | -203.53 → -13.57 | 门控→全高度：1次切换（v1距离门控→v6 exp门控但参数不同，本质仍为门控） |
| 完成/成功代理 | v2_landing(主导) → v7_landing_completion(主导) | -37.03 → 146.32 | 连续奖励→稀疏bonus：1次切换（v2 proximity连续→v7 双足接触稀疏） |

**方向切换汇总**：
- 着陆/接触信号：5次切换（在该功能上的设计振荡最显著）
- 连续奖励 vs 稀疏bonus 在这 5 次中交替出现，v2(连续)→v3(稀疏)→v5(连续)→v6(稀疏)→v7(稀疏)

**注**：v4 的 velocity_guidance 未归入上述任何语义功能——它属于"速度方向引导"这一独立功能，仅在 v4 出现一次后即被移除。