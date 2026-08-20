# 训练数据报告 — Iter 07

---

## 1. 行为事实

- **得分**: score=255.28, range=[164.97, 299.71]
- **终止方式**: terminated=18/20 (成功), truncated=2/20 (时间上限)
- **回合长度**: len=351.65
- **Checkpoint 轨迹**: 
  - 40%: score_mean=0.00
  - 80%: score_mean=0.00
  - (注：仅提供 40% 和 80% 两个 checkpoint 数据点)

---

## 2. 骨架演进表

| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active | contact_reward_ep_sum | contact_reward_active | landing_comfort_ep_sum | landing_comfort_active | progress_ep_sum | progress_active | contact_ep_sum | contact_active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -214.16 | 182.50 | 20/0/0 | 0.152 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | — | — | — | — | — | — | — | — | — | — |
| 2 | -109.58 | 68.45 | 20/0/0 | 1.120 | 100.0% | -19.091 | 100.0% | 0.000 | 0.0% | — | — | — | — | — | — | — | — | — | — |
| 3 | -115.21 | 68.35 | 20/0/0 | 1.119 | 100.0% | -20.744 | 100.0% | — | — | 25.000 | 0.4% | — | — | — | — | — | — | — | — |
| 4 | -111.06 | 68.40 | 20/0/0 | 1.120 | 100.0% | -19.297 | 100.0% | — | — | — | — | 1.150 | 1.7% | — | — | — | — | — | — |
| 5 | -119.00 | 68.40 | 20/0/0 | 1.117 | 100.0% | -20.641 | 100.0% | — | — | 25.000 | 0.4% | — | — | — | — | — | — | — | — |
| 6 | -108.18 | 68.45 | 20/0/0 | 1.120 | 100.0% | -19.127 | 100.0% | — | — | — | — | — | — | 0.428 | 0.7% | — | — | — | — |
| **7** | **255.28** | **351.65** | **18/0/2** | — | — | — | — | — | — | — | — | — | — | — | — | 0.843 | 100.0% | 25.830 | 36.7% |

---

## 3. 数字标记

### 主导
- **contact**: magnitude_share=96.4% > 80%，为主导组件。

### 死组件
- 无。progress active_rate=100.0%，contact active_rate=36.7%，均不低于 2%。

### 淹没
- **progress**: active_rate=100.0% > 90%，但 magnitude_share=3.6% < 3%（接近但未低于 3% 阈值；标记为边界情况，不正式判定为淹没）。

### 全覆盖惩罚
- 无。contact 的 signed_share 和 magnitude_share 均为正。

### 量级对比（每步均值 = ep_sum ÷ len）
- contact: 25.830 / 351.65 ≈ **0.0735**
- progress: 0.843 / 351.65 ≈ **0.0024**

本迭代仅有两个组件，contact 每步均值最大，progress 每步均值最小。

### 修改变化（相对于 Iter 06）
Iter 07 的组件体系与 Iter 06 完全不同（从 `goal_approach + soft_landing + landing_comfort` 重构为 `progress + contact`），无法直接逐组件对比 ep_sum 和 active_rate 的 Δ。

---

## 4. 语义演化（跨轮组件功能追踪）

### 功能分组分析

#### 1. 着陆/接触信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | crash_avoidance | 稀疏惩罚 (一步 -10) | 负 | 0.000 | 0.0% |
| 2 | crash_avoidance | 稀疏惩罚 (一步 -10) | 负 | 0.000 | 0.0% |
| 3 | landing_success | 稀疏bonus (一步 +100, 严条件) | 正 | 25.000 | 0.4% |
| 4 | contact_reward | 连续奖励 (每步 +1, 仅双腿接触) | 正 | 1.150 | 1.7% |
| 5 | landing_success | 稀疏bonus (一步 +100, 宽条件) | 正 | 25.000 | 0.4% |
| 6 | landing_comfort | 连续奖励 (每步 max+1, 稳定性缩放) | 正 | 0.428 | 0.7% |
| **7** | **contact** | **连续奖励 (每步 +0.2, 仅双腿接触)** | **正** | **25.830** | **36.7%** |

**方向切换**：稀疏bonus ↔ 连续奖励：4 次切换（Iter 3 landing_success 稀疏 → Iter 4 contact_reward 连续 → Iter 5 landing_success 稀疏 → Iter 6 landing_comfort 连续）。Iter 7 维持连续奖励形态，但增加 gated progress 作为主导前导信号后，contact 的 active_rate 从 ~1% 跃升至 36.7%。

#### 2. 接近/进度信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | goal_approach | 连续 (delta 距离) | 正 | 0.152 | 100.0% |
| 2 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| 3 | goal_approach | 连续 (delta 距离) | 正 | 1.119 | 100.0% |
| 4 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| 5 | goal_approach | 连续 (delta 距离) | 正 | 1.117 | 100.0% |
| 6 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| **7** | **progress** | **连续 (delta 距离, health-gated)** | **正** | **0.843** | **100.0%** |

**方向切换**：Iter 1-6 均为纯 delta 距离形态；Iter 7 引入指数 health gate（angle_gate × vel_gate × angvel_gate），仅在 progress>0 时衰减。形态上从"纯改善量"转向"门控改善量"：1 次切换。ep_sum 从 ~1.12 降至 0.843，但配合 contact 激活后成功实现 task solved。

#### 3. 稳定性/约束信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | soft_landing | 低空门控二次惩罚 | 负 | 0.000 | 0.0% |
| 2 | soft_landing | 全高度线性惩罚 | 负 | -19.091 | 100.0% |
| 3 | soft_landing | 全高度线性惩罚 | 负 | -20.744 | 100.0% |
| 4 | soft_landing | 全高度线性惩罚 | 负 | -19.297 | 100.0% |
| 5 | soft_landing | 全高度线性惩罚 | 负 | -20.641 | 100.0% |
| 6 | soft_landing | 全高度线性惩罚 | 负 | -19.127 | 100.0% |
| **7** | — | **无独立稳定性组件** | — | — | — |

**方向切换**：Iter 1 门控二次 → Iter 2 全高度线性：1 次切换。Iter 7 完全移除了独立稳定性惩罚组件，改用 health gate 内嵌于 progress 中隐式约束 stability（通过 angle_gate/vel_gate/angvel_gate 衰减正向进展）。这意味着稳定性功能从"独立显式惩罚"切换为"隐式门控"：1 次切换。

---

### 总结
Iter 07 通过将历史迭代中的三大功能（接近、着陆、稳定性）压缩为两个组件（progress + contact），实现了 score 从负值到 255.28 的跨越。稳定性功能不再独立存在，而是被吸收进 progress 的 exponential health gate 中。着陆信号从迭代 3-6 的多次稀疏/连续振荡中稳定为低值连续奖励（0.2/step），配合 gated progress 后 active_rate 显著提升。