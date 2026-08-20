# 训练数据报告

## 1. 行为事实
- 本轮得分：score=-13.574，range=[-45.505, 26.677]
- 终止方式：terminated 0/20（无成功着陆），truncated 20/20（全部达到时间上限）
- 回合长度：len=1000.000（全部集数跑满 1000 步）
- 官方得分Checkpoint轨迹（每10%步数）：-5.3 → 25.9 → 10.9 → -54.7 → -15.2 → -4.2 → 38.6 → 164.0 → -5.1 → 106.7
  - 总体趋势：波动剧烈，cp80%达到最高点164.0，之后回落
- 组件活跃率：distance_progress 100.0%，fuel_penalty 98.9%，stability 100.0%，landing_completion 0.0%

## 2. 骨架演进表
| iter | score | len | term | progress/distance_progress ep_sum | progress/distance_progress active | speed_penalty/landing ep_sum | speed_penalty/landing active | angular_vel_penalty/stability ep_sum | angular_vel_penalty/stability active | landing/landing_completion/fuel_penalty ep_sum | landing/landing_completion/fuel_penalty active | velocity_guidance ep_sum | velocity_guidance active |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | -203.531 | 185.800 | 20/0 | -0.200 | 100.0% | -0.669 | 15.5% | -0.434 | 100.0% | 0.000 | 0.0% | — | — |
| 2 | -37.031 | 962.550 | 4/16 | 0.665 | 100.0% | -42.130 | 100.0% | -2.457 | 100.0% | 697.436 | 99.6% | — | — |
| 3 | -222.850 | 209.600 | 20/0 | -0.299 | 100.0% | -0.735 | 14.8% | -0.368 | 100.0% | 0.000 | 0.0% | — | — |
| 4 | -295.327 | 138.900 | 20/0 | -0.026 | 100.0% | -2.097 | 63.2% | -1.266 | 100.0% | — | — | -1.596 | 100.0% |
| 5 | -197.455 | 185.500 | 20/0 | -0.183 | 100.0% | -0.717 | 15.8% | -0.343 | 100.0% | 0.000 | 0.0% | — | — |
| 6 | -13.574 | 1000.000 | 0/20 | 6.331 | 100.0% | — | — | -1.416 | 100.0% | -9.889(fuel) / 0.000(landing) | 98.9%(fuel) / 0.0%(landing) | — | — |

注：iter 1-5 组件名为 progress, speed_penalty, angular_vel_penalty, landing, velocity_guidance；iter 6 重新命名为 distance_progress, stability, landing_completion, fuel_penalty。表中按语义列对齐，同名显示为合并列。

## 3. 数字标记

### 死组件
- **landing_completion**：active_rate=0.0%（<2%），ep_sum_mean=0.000，在所有20集评估中从未激活

### 全覆盖惩罚
- 无。fuel_penalty active_rate=98.9%（>95%）但 signed_share 为负（-53.2%），检查：active_rate 98.9% > 95%，signed_share -53.2% < 0 → **fuel_penalty 全覆盖惩罚**（每步几乎都触发，且全部为负贡献）

### 主导
- **fuel_penalty**：magnitude_share=53.2%（>80%？否，53.2%不满足>80%条件）
- 无组件 magnitude_share > 80%

### 量级对比（每步均值 ep_sum ÷ len）
- 最大两个：
  - distance_progress：6.331 ÷ 1000 = 0.00633
  - stability：|-1.416| ÷ 1000 = 0.00142 → 实际第二大为 fuel_penalty：|-9.889| ÷ 1000 = 0.00989
  - 修正：fuel_penalty 每步均值 0.00989，distance_progress 每步均值 0.00633，stability 每步均值 0.00142，landing_completion 每步均值 0.00000
  - 最大两个：fuel_penalty (0.00989), distance_progress (0.00633)
  - 最小两个：landing_completion (0.00000), stability (0.00142)

### 修改变化（iter 5 → iter 6，按语义对应）
本报告不做组件映射推断，仅提供原始数据对比。iter 6 重命名了全部四个组件，无法建立一一对应关系。

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 进度信号 | v1_progress → v2_progress → v3_progress → v4_progress → v5_progress → v6_distance_progress | -0.200 → 0.665 → -0.299 → -0.026 → -0.183 → 6.331 | 缩放因子×5：iter 6 引入（旧版均为×1.0） |

- v1-v5：progress = old_dist - new_dist（状态值，×1.0），始终在 [-0.3, 0.7] 范围内
- v6：distance_progress = (old_dist - new_dist) × 5.0（状态值，×5.0），ep_sum 从 ~0.7 跳升至 6.331
- 方向切换：无（始终为状态值/改善量形式），但缩放因子从 1.0 → 5.0

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v1_landing(死) → v2_landing(主导) → v3_landing(死) → v5_landing(死) → v6_landing_completion(死) | 0.000 → 697.436 → 0.000 → 0.000 → 0.000 | contact-gated↔proximity-gated：4次切换 |
| 燃料成本 | v6_fuel_penalty | -9.889 | 新组件（iter 6 首次引入） |

着陆信号详细演变：
- v1：contact-gated + quality factors，权重2.0 → 死组件（active_rate 0.0%）
- v2：proximity-gated + sum-of-factors + contact_score，权重3.0 → 主导组件（ep_sum 697.4，magnitude_share 93.8%），agent 学会 exploit 悬停
- v3：contact-gated + quality factors（无 proximity 门控），权重5.0 → 死组件（active_rate 0.0%）
- v4：移除 landing，替换为 velocity_guidance
- v5：proximity-gated + contact_multiplier（1x/3x/5x），权重0.5 → 死组件（active_rate 0.0%）
- v6：contact-gated（both_feet > 0.5）+ 条件性乘积奖励，权重2.0 → 死组件（active_rate 0.0%）

方向切换（不同设计范式之间的切换）：
- 稀疏bonus（contact-gated）↔ 连续奖励（proximity-gated）：v2（连续proximity）→ v3（稀疏contact）→ v5（连续proximity）→ v6（稀疏contact）= **3次切换**
- v1→v2：从 contact-gated 切换到 proximity-gated（第1次）
- v2→v3：从 proximity-gated 切换到 contact-gated（第2次）
- v3→v5：从 contact-gated 切换到 proximity-gated（第3次）
- v5→v6：从 proximity-gated 切换到 contact-gated（第4次）
- 总计：4次切换

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 速度惩罚 | v1_speed_penalty → v2_speed_penalty → v3_speed_penalty → v4_speed_penalty → v5_speed_penalty | -0.669 → -42.130 → -0.735 → -2.097 → -0.717 | 门控位置：3次切换（closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist → closeness-on-new_dist，无切换） |
| 角速度惩罚 | v1_angular_vel_penalty → v2_angular_vel_penalty → v3_angular_vel_penalty → v4_angular_vel_penalty → v5_angular_vel_penalty | -0.434 → -2.457 → -0.368 → -1.266 → -0.343 | 始终为全高度 |w·abs(ang_vel)| |
| 综合稳定性 | v6_stability | -1.416 | 门控：exp(-2d)，形式从线性变为二次+门控 |

v1-v5：速度惩罚 + 角速度惩罚 分离为两个组件
- speed_penalty：-w_speed × closeness × speed，closeness = max(0, 1 - new_dist/1.5)
- angular_vel_penalty：-w_angvel × |next_ang_vel|，始终激活

v6：合并为 stability，形式变为 -(w_vel×velocity² + w_angle×angle² + w_angv×ang_vel²) × exp(-2d)
- 从基于 next_state 切换到基于当前 state（obs而非next_obs）
- 惩罚从线性转为二次项
- 门控从分段线性（closeness）切换到指数衰减（exp(-2d)）
- 方向切换：速度惩罚从"线性门控"切换到"二次指数门控" = 1次切换

### 完成/成功代理
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 导航引导 | v4_velocity_guidance | -1.596 | 仅存在于 iter 4 |
| 着陆完成 | v6_landing_completion | 0.000 | 条件性乘积形式，仅激活于 both_feet 接触 |

- v4 velocity_guidance：奖励速度方向与目标方向的对齐度（余弦相似度 × 距离衰减），ep_sum=-1.596（agent 未能学会利用）
- v6 landing_completion：contact-gated 完成奖金（both_feet + close_to_centre），乘积形式（speed_quality × angle_quality × angvel_quality），ep_sum=0.000（全程未触发）
- 设计从"连续导航引导"切换到"稀疏完成奖金" = 1次切换