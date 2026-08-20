# 设计理由

## 第 0 步：信号覆盖审计

- **终止模式分析**：20/20 terminated, 0/20 truncated, len=438.45, score=237.90。无 early_terminal（<150 步且 score<-50）。agent 稳定着陆并触发成功终止（`body_not_awake_or_settled`），目标已超额达成（gap=-37.9）。
- **观测使用扫描**：全部 8 个观测维度均在代码中使用。无缺失信号。
- **信号缺口判断**：**信号齐全但校准问题**——问题不在遗漏信号，而在于 zombie 组件和主信号的权重分布。
- **僵尸组件**：`angular_velocity_penalty` 的 `active_rate=95.4%` 但 `episode_sum_mean=-0.016`（幅度占比 0.0%）。该组件对训练几乎无贡献，属于僵尸组件，应被替换。

## 第 1 步：行为诊断

| iter | 关键变化 | len | score |
|---|---|---|---|
| 6 | 含 `angle_penalty`（绝对值形式）+ `contact_stability` + `success_bonus` | 386.15 | **243.15** |
| 7 | 移除 `angle_penalty`，新增 `angular_velocity_penalty` | 438.45 | 237.90 |

- **agent 在做什么**：平稳飞行至目标区域，减速并以良好姿态着陆。iter 7 比 iter 6 **慢 52 步**且**低 5.25 分**。
- **根因推断**：`angle_penalty` → `angular_velocity_penalty` 的替换产生了反效果。惩罚角速度（旋转速率）与惩罚绝对角度偏移是不同的梯度信号——着陆需要的是**最终角度趋近于零**，而非仅消除旋转。`angle_penalty` 为直立姿态提供了直接梯度，有助于更快稳定，减少振荡。
- **方向判断**：iter 7 的预判为 ❓（不确定），但实际表现为负面。累积记录无连续 ❌，方向仍可修复。

## 第 2 步：选择干预层级 — Level 2（结构变换）

替换僵尸组件 `angular_velocity_penalty` 为 **有阈值的 angle hinge penalty**：

| 证据 | 变换 | 理由 |
|---|---|---|
| `angular_velocity_penalty` 幅度占比 0.0%，active_rate 95.4% | 删除该组件 | 僵尸组件，占位但不产生有效梯度 |
| iter 6 的 `angle_penalty`（绝对值）优于 iter 7 | 恢复角度惩罚，但改用 hinge 形态 | 直接约束绝对角度，且阈值化避免在正常飞行中激活 |
| 当前无界二次惩罚（`angvel_n²`）在数值上塌缩 | 二值/塌缩 → hinge bounded | hinge 在阈值以下为零，超过后以二次增长，提供清晰梯度 |

**新组件设计**：`angle_deviation = max(0, abs(angle_n) - ANGLE_HINGE_THRESHOLD)`，惩罚 = `-weight * angle_deviation²`。

## 设计校准

1. **阈值设定**：`ANGLE_HINGE_THRESHOLD = 0.25`。成功着陆的角度阈值为 `0.2`（见 `SUCCESS_ANGLE_THRESHOLD`），hinge 设在终止边界的 125% 处，留出飞行机动空间。
2. **系数校准**：
   - 主信号 per-step ≈ 35.23 / 438 ≈ 0.080（success_bonus）
   - 目标惩罚 per-step ≤ 0.3 × 0.080 = 0.024
   - 在 \|angle\|=0.3 时：penalty = -0.05 × 0.05² = -0.000125 → 可忽略
   - 在 \|angle\|=0.5 时：penalty = -0.05 × 0.25² = -0.003125 → 正常
   - 在 \|angle\|=1.0 时：penalty = -0.05 × 0.75² = -0.028125 → 极端但罕见 ✓
3. **总惩罚负担**：仅此一项惩罚，per-step ≤ 0.003（正常飞行）远低于 0.04 上限 ✓
4. **gate 不塌缩**：在"不理想但安全"区域（\|angle\|≈0.3）惩罚 ≈ 0 → 不干扰飞行，超过阈值后渐进增强。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: replace zombie angular_velocity_penalty with angle_hinge_penalty.
        Hinge threshold at 0.25 rad — no penalty in normal flight maneuvering,
        quadratic penalty for larger angular deviations to encourage upright stability.
        Evidence: iter6 (with angle_penalty) len=386 score=243 > iter7 len=438 score=238.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.05
    ANGLE_HINGE_THRESHOLD = 0.25
    ANGLE_HINGE_WEIGHT = 0.05
    CONTACT_WEIGHT = 0.1
    PROXIMITY_THRESHOLD = 0.5
    SUCCESS_DIST_THRESHOLD = 0.3
    SUCCESS_SPEED_THRESHOLD = 0.3
    SUCCESS_ANGLE_THRESHOLD = 0.2
    SUCCESS_SCALE = 1.0

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    speed = (x_v_n ** 2 + y_v_n ** 2) ** 0.5
    speed_bonus = 1.0 / (1.0 + speed)
    soft_landing = LANDING_WEIGHT * proximity * speed_bonus

    # ---------- 3) angle hinge penalty (replaces angular_velocity_penalty) ----------
    angle_deviation = max(0.0, abs(angle_n) - ANGLE_HINGE_THRESHOLD)
    angle_hinge_penalty = -ANGLE_HINGE_WEIGHT * (angle_deviation ** 2)

    # ---------- 4) contact stability reward ----------
    contact_flag = max(left_n, right_n)
    angle_bonus = 1.0 / (1.0 + abs(angle_n))
    contact_stability = (
        CONTACT_WEIGHT * proximity * contact_flag * speed_bonus * angle_bonus
    )

    # ---------- 5) success bonus (dense continuous factor) ----------
    proximity_factor = max(0.0, 1.0 - R_next / SUCCESS_DIST_THRESHOLD)
    speed_factor = max(0.0, 1.0 - speed / SUCCESS_SPEED_THRESHOLD)
    angle_factor = max(0.0, 1.0 - abs(angle_n) / SUCCESS_ANGLE_THRESHOLD)
    contact_factor = max(left_n, right_n)
    success_factor = (proximity_factor * speed_factor *
                      angle_factor * contact_factor)
    success_bonus = SUCCESS_SCALE * success_factor

    # ---------- aggregate ----------
    total_reward = (
        progress_reward + soft_landing + angle_hinge_penalty +
        contact_stability + success_bonus
    )

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_hinge_penalty": angle_hinge_penalty,
        "contact_stability": contact_stability,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 全部 8 个观测维度已使用；`angular_velocity_penalty` 为 zombie（active 95% 但贡献 ≈0），需替换
- **behavior**: agent 稳定着陆但 iter 7 比 iter 6 慢 52 步且低 5 分，角速度惩罚未带来改善
- **signal**: 缺少对绝对角度的直接约束；iter 6 的 `angle_penalty` 提供了更有效的直立梯度
- **level**: Level 2 — 结构变换：将塌缩的二次角速度惩罚替换为带阈值的 hinge 角度惩罚
- **hypothesis**: hinge 角度惩罚将在正常飞行（|angle|<0.25）中不激活，在角度过大时提供清晰梯度，帮助 agent 更快恢复直立姿态，减少振荡时间，改善 iter 6 的性能或至少恢复至其水平
- **risk**: 若阈值 0.25 对飞行机动过于宽松，则惩罚可能激活不足（类似 zombie）；但 0.05 的权重确保一旦超过阈值即有可感知梯度