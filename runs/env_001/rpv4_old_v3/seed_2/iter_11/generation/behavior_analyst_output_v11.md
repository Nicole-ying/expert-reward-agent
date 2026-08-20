# 训练数据报告 — iter_10

---

## 1. 行为事实

- 得分：score=131.35，range=[12.95, 257.37]
- 终止方式：terminated=11/20（成功着陆），truncated=9/20（超时）
- 回合长度：len=831.25
- Checkpoint 轨迹：
  - cp_40%: official_score=0.00（std=0.00）
  - cp_80%: official_score=0.00（std=0.00）
  - 两个checkpoint得分均为0，说明训练过程中间阶段官方得分无变化

---

## 2. 骨架演进表

仅列出有完整组件分解数据的轮次（iter_01 — iter_10）。

> 组件命名已统一为最近两轮使用的名称。`progress_shaping` 与 `progress` 为同一语义功能的不同名称，表中以 `progress` 统一标记。

| iter | score | len | term | progress_ep_sum | progress_active | vel_damping_ep_sum | vel_damping_active | angle_stab_ep_sum | angle_stab_active | fuel_penalty_ep_sum | fuel_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | crash_penalty_ep_sum | crash_penalty_active |
|------|-------|-----|------|----------------|-----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|-----------------------------|---------------------------|-------------------------------|-------------------------------|----------------------|---------------------|
| 1 | -155.98 | 154.40 | 20/0/0 | 1.582 | 100.0% | — | — | -1.141* | 100.0% | -1.804* | 58.4% | — | — | -0.067 | 5.3% | — | — |
| 2 | -119.51 | 68.30 | 20/0/0 | 5.583 | 100.0% | — | — | -0.205 | 100.0% | -0.071* | 5.2% | 5.000 | 0.1% | — | — | — | — |
| 3 | -203.45 | 154.15 | 20/0/0 | 1.488 | 100.0% | — | — | -2.460 | 100.0% | -1.472* | 47.7% | 0.000 | 0.0% | — | — | 0.000 | 0.0% |
| 4 | -114.38 | 68.45 | 20/0/0 | 5.597 | 100.0% | — | — | -0.087 | 100.0% | -0.097* | 7.1% | 30.000 | 0.4% | — | — | — | — |
| 5 | -46.51 | 740.15 | 7/13/0 | 5.105 | 99.9% | — | — | -6.828 | 100.0% | -13.469* | 91.0% | 1195.000 | 1.6% | — | — | — | — |
| 6 | -111.18 | 68.45 | 20/0/0 | 5.603 | 100.0% | — | — | -0.020 | 100.0% | -0.067* | 4.9% | 0.000 | 0.0% | — | — | — | — |
| 7 | -98.93 | 69.60 | 20/0/0 | 5.553 | 100.0% | — | — | -0.221 | 100.0% | -0.178* | 12.8% | 5.000 | 0.1% | — | — | — | — |
| 8 | 26.45 | 206.50 | 20/0/0 | 2.667 | 100.0% | -1.577 | 99.9% | -0.263 | 100.0% | -1.232 | 59.6% | — | — | — | — | — | — |
| 9 | 145.17 | 718.80 | 9/11/0 | 2.638 | 99.2% | -2.022 | 99.2% | -0.748 | 100.0% | -4.268 | 59.4% | 5330.000 | 7.4% | — | — | — | — |
| **10** | **131.35** | **831.25** | **11/9/0** | **24.704** | **100.0%** | **-1.529** | **99.9%** | **-0.564** | **100.0%** | **-7.934** | **95.4%** | **6100.000** | **7.3%** | **—** | **—** | **—** | **—** |

> `term` 列格式：`terminated/truncated/early_terminal`。
> `*` 标记：iter_01 中的 `angle_penalty` 和 `action_cost` 在概念上对应后续的 `angle_stab` 和 `fuel_penalty`，但计算方式不同，仅供参考。
> `action_cost` 在 iter_01—07 中作为独立组件存在，iter_08 起改为 `fuel_penalty`（语义上等价）。

---

## 3. 数字标记

### 主导
- **success_landing_bonus**：magnitude_share=99.3%，为绝对主导组件（ep_sum=6100.00）

### 死组件
- 无。最低 active_rate 为 success_landing_bonus (7.3%)，但这是稀疏事件奖励的预期行为，实际非零。

### 淹没
- **angle_stab**：active_rate=100.0%，magnitude_share≈0.0%（ep_sum_mean=-0.564，被 success_landing_bonus 的 6100.00 完全淹没）
- **vel_damping**：active_rate=99.9%，magnitude_share≈0.0%（ep_sum_mean=-1.529，同上）
- **progress**：active_rate=100.0%，magnitude_share=0.5%（ep_sum_mean=24.704 vs 6100.00）

### 全覆盖惩罚
- 无。所有惩罚组件（vel_damping、angle_stab、fuel_penalty）的 signed_share 均为负但 magnitude_share 极低（0.0%—0.1%），不满足 active_rate>95% 的正 magnitude_share 条件。

### 量级对比（每步均值 = ep_sum ÷ len）
- 最大：
  1. success_landing_bonus: 6100.00 / 831.25 ≈ 7.338
  2. progress: 24.704 / 831.25 ≈ 0.030
- 最小：
  1. angle_stab: -0.564 / 831.25 ≈ -0.00068
  2. vel_damping: -1.529 / 831.25 ≈ -0.00184

所有 shaping/constraint 组件（progress、vel_damping、angle_stab、fuel_penalty）的每步均值均在 0.03 以下，而 success_landing_bonus 约为 7.34，相差约 245 倍。

### 修改变化（iter_09 → iter_10）
| component | Δ ep_sum | Δ active_rate |
|-----------|----------|---------------|
| progress | +22.066 | +0.8 pp |
| vel_damping | +0.492 | +0.7 pp |
| angle_stab | +0.184 | 0.0 pp |
| fuel_penalty | -3.666 | +36.0 pp |
| success_landing_bonus | +770.0 | -0.1 pp |

唯一显著变化：progress 的 ep_sum 从 2.638 升至 24.704（约 9.4 倍），对应 w_progress 从 2.0 提升至 20.0。fuel_penalty 的 active_rate 从 59.4% 跳升至 95.4%，ep_sum 翻倍。success_landing_bonus ep_sum 增加 770（≈14%）。

---

## 4. 语义演化（跨轮组件功能追踪）

### 着陆/接触信号（success landing bonus / landing success）
| iter | 组件名 | ep_sum_mean | active_rate | 阈值 (dist/speed/tilt) | 触发方式 |
|------|--------|-------------|-------------|---------------------|---------|
| 2 | success_landing_bonus | 5.000 | 0.1% | 0.1/0.2/0.1 | any_contact 持续判断 |
| 3 | success_landing_bonus | 0.000 | 0.0% | 0.1/0.2/0.1 | any_contact 持续判断 |
| 4 | success_landing_bonus | 30.000 | 0.4% | 0.2/0.5/0.2 | any_contact 持续判断 |
| 5 | success_landing_bonus | 1195.000 | 1.6% | 0.4/1.0/0.5 | any_contact 持续判断 |
| 6 | success_landing_bonus | 0.000 | 0.0% | 0.3/0.3/0.3 | any_contact 转变事件 |
| 7 | success_landing_bonus | 5.000 | 0.1% | 0.4/1.0/0.5 | any_contact 转变事件 |
| 9 | success_landing_bonus | 5330.000 | 7.4% | 0.4/1.0/0.5 | both_contact 转变事件 |
| 10 | success_landing_bonus | 6100.000 | 7.3% | 0.4/1.0/0.5 | both_contact 转变事件 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v2_success_landing_bonus → v3_success_landing_bonus → v4_success_landing_bonus → v5_success_landing_bonus → v6_success_landing_bonus → v7_success_landing_bonus → v9_success_landing_bonus → v10_success_landing_bonus | -120 → -203 → -114 → -47 → -111 → -99 → 145 → 131 | 持续性→转变事件：1次切换（v6）；阈值收紧↔放宽：4次切换（v4放宽→v6收紧→v7放宽→v9不变）；接触条件：any→both：1次切换（v9） |

---

### 接近/进度信号（progress/progress_shaping）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 | 数学形式 |
|------|--------|-------------|-------------|------|---------|
| 1 | progress_shaping | 1.582 | 100.0% | 5.0 | 势能差：5×((-next_dist)-(-dist)) |
| 2 | progress_shaping | 5.583 | 100.0% | 5.0 | 同上 |
| 3 | progress_shaping | 1.488 | 100.0% | 5.0 | 同上 |
| 4 | progress_shaping | 5.597 | 100.0% | 5.0 | 同上 |
| 5 | progress_shaping | 5.105 | 99.9% | 5.0 | 同上 |
| 6 | progress_shaping | 5.603 | 100.0% | 5.0 | 同上 |
| 7 | progress_shaping | 5.553 | 100.0% | 5.0 | 同上 |
| 8 | progress | 2.667 | 100.0% | 2.0 | 距离减少量：dist-next_dist |
| 9 | progress | 2.638 | 99.2% | 2.0 | 同上 |
| 10 | progress | 24.704 | 100.0% | 20.0 | 同上 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近信号 | v1_progress_shaping → v2_progress_shaping → ... → v7_progress_shaping → v8_progress → v9_progress → v10_progress | -156 → -120 → -203 → -114 → -47 → -111 → -99 → 26 → 145 → 131 | 势能差→距离减少量：1次切换（v8）；权重调整：5.0→2.0→20.0，2次切换 |

---

### 着陆速度约束
| iter | 组件名 | ep_sum_mean | active_rate | 形式 |
|------|--------|-------------|-------------|------|
| 1 | landing_velocity_penalty | -0.067 | 5.3% | 门控 hinge（y<0.5） |
| 2—10 | — | — | — | 移除，未恢复 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆速度约束 | v1_landing_velocity_penalty → (删除) | -156 | 仅存1轮后删除 |

---

### 稳定性/约束信号（angle_stab/angle_penalty + vel_damping）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 | 形式 |
|------|--------|-------------|-------------|------|------|
| 1 | angle_penalty | -1.141 | 100.0% | -0.5 | body_angle² |
| 2—7 | angle_penalty | -0.087~-6.828 | 100.0% | -0.5 | body_angle² |
| 8—10 | angle_stab | -0.263~-0.748 | 100.0% | -0.05 | angle²+angvel²（next_obs） |
| 8—10 | vel_damping | -1.529~-2.022 | 99.2%—99.9% | -0.05 | vx²+vy²（next_obs） |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 稳定性 | v1_angle_penalty → v8_angle_stab+vel_damping | -156 → 26 | obs角度→next_obs全状态：1次切换（v8）；单组件→双组件：1次切换（v8） |

---

### 效率/惩罚信号（fuel_penalty/action_cost）
| iter | 组件名 | ep_sum_mean | active_rate | 权重 |
|------|--------|-------------|-------------|------|
| 1—7 | action_cost | -0.067~-13.469 | 4.9%—91.0% | -0.02 |
| 8—10 | fuel_penalty | -1.232~-7.934 | 59.4%—95.4% | -0.01 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 燃料效率 | v1_action_cost → v8_fuel_penalty | -156 → 26 | 权重-0.02→-0.01：1次切换 |

---

### 坠毁惩罚
| iter | 组件名 | ep_sum_mean | active_rate |
|------|--------|-------------|-------------|
| 3 | crash_penalty | 0.000 | 0.0% |
| 4—10 | — | — | 删除 |

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 失败惩罚 | v3_crash_penalty → (删除) | -203 | 仅存1轮，死组件即删除 |

---

**方向切换汇总**：
- 着陆信号：持续↔转变 1次，阈值收紧↔放宽 4次，接触条件 1次
- 接近信号：势能差→距离减少量 1次，权重 2次
- 稳定性：obs→next_obs 1次，单组件→双组件 1次
- 燃料效率：权重调整 1次

≥2次切换的功能：着陆信号（6次方向切换，其中阈值收紧↔放宽 4次），接近信号（3次）。