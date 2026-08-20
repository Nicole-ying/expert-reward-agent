## 诊断

**Agent 状态**：全部 episode 在 72 步 crash（100% early termination），得分 -108。

**根因分析**（从 ratio_to_progress 看）：
| 组件 | ratio | 问题 |
|------|-------|------|
| stability_penalty | **-5.09** | 是 progress 信号的 5 倍 |
| distance_penalty | **-3.02** | 是 progress 信号的 3 倍 |
| soft_landing_bonus | +0.033 | nonzero_rate **0.05%**，几乎从不触发 |
| **合计** | **-8:1** | 惩罚淹没了主学习信号 |

这正是 `stability_penalty_dominance` + `early_failure_or_crash` 的经典模式：agent 被惩罚压得无法行动，选择"不动/早 crash"来避免惩罚。

**验证失败原因**：奖励函数中惩罚信号是主信号的 8 倍，且 landing bonus 无梯度（二值条件），属于不合格设计。

## 修订方案（层次 1 + 2）

1. **稳定性惩罚**：系数削减 ~10x，从 `0.05/0.2` 降到 `0.005/0.02`
2. **距离惩罚**：从 `-0.05` 降到 `-0.005`（同样 ~10x）
3. **软着陆 bonus**：从二值 if 条件改为**连续乘积 proxy**（`bounded_continuous_proxy` 技法），让 agent 在接近着陆条件时获得梯度信号

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (anchor, drastically reduced) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (drastically reduced: 10x smaller) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- continuous soft landing proxy (replaces binary bonus) ----
    # Use bounded continuous factors so agent gets gradient as it approaches
    # a good landing, not just at the exact moment of contact.
    LANDING_DIST_THRESH = 1.0
    VEL_THRESH = 0.5
    ANGLE_THRESH = 0.3

    proximity_factor = max(0.0, 1.0 - d_next / LANDING_DIST_THRESH)
    vel_x_factor = max(0.0, 1.0 - abs(vx) / VEL_THRESH)
    vel_y_factor = max(0.0, 1.0 - abs(vy) / VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Only reward when both legs are on the ground (contact), but use
    # continuous factors so the approach path also gets shaped.
    both_legs_contact = 1.0 if (next_obs[6] == 1.0 and next_obs[7] == 1.0) else 0.0

    landing_bonus = (
        proximity_factor
        * vel_x_factor
        * vel_y_factor
        * angle_factor
        * both_legs_contact
        * 2.0  # scale factor so a perfect landing gives meaningful reward
    )

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + landing_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": landing_bonus,
        "total_reward": total,
    }
    return float(total), components
```

## 改动说明

| 改动 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| distance_penalty 系数 | `-0.05` | `-0.005` | 削减 10x，原值 ratio 达 -3.02 |
| stability_penalty vx/vy 系数 | `-0.05` | `-0.005` | 削减 10x |
| stability_penalty angle/ang_vel 系数 | `-0.2` | `-0.02` | 削减 10x |
| soft_landing_bonus 形式 | 二值 if 条件 | 连续乘积 `max(0, 1-x/D)` | 提供梯度，nonzero_rate 从 0.05% → 应显著提升 |
| soft_landing_bonus 权重 | `1.0` | `2.0` | 配合连续形式，完美着陆时约 2.0 |
| LANDING_DIST_THRESH | `0.5` | `1.0` | 放宽距离阈值，让更多状态进入奖励区 |
| VEL_THRESH | `0.2` | `0.5` | 放宽速度阈值 |
| ANGLE_THRESH | `0.1` | `0.3` | 放宽角度阈值 |

预期效果：progress_delta 成为主导信号（ratio≈1.0），penalty 总和 ratio < 1.0，landing_bonus 提供有梯度的着陆引导。