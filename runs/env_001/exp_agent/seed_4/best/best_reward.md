## 诊断

### 1. Agent 发生了什么？

得分 242 → **新最佳**，episode 长度 624.5 → agent 没有 crash（非短 episode），但也没有跑满 1000 步。它在中等长度内完成了任务（可能是着陆）。从 iter 5（-54, crash）到 iter 6（242）的跳跃说明恢复 w_landing=0.03 是正确方向。

### 2. 哪个组件是主要原因？

核心问题：**stability_penalty 的量级几乎等于 progress 信号**。

| 组件 | ratio_to_progress |
|------|------------------|
| progress_delta_reward | 1.00 |
| stability_penalty | **-0.85** |
| soft_landing_proxy | 11.02 |

`stability_penalty` 绝对值 ratio 为 0.85，远超原则 1 的 0.50 阈值。这意味着 agent 每向目标前进一步获得的 progress 奖励（~0.002），几乎被稳定性惩罚（~-0.0019）完全抵消。agent 被迫在"前进"和"保持稳定"之间做艰难权衡，导致动作保守、episode 偏长（624 步而非更高效）。

landing proxy ratio=11 是 bonus 类信号的正常表现（nonzero_rate 99%，密集梯度），不是问题。

### 3. 我之前改了什么？

上轮（iter 6）：w_landing 从 0.01 → 0.03，恢复到 iter 3 的配置。效果：从 crash（-54）恢复到新最佳（242）。landing proxy 是目前的主要驱动力，**不要动它**。

### 修改决策

**层次 1：改系数。** 将 stability penalty 的三个权重各降低约 3 倍，使 penalty ratio 从 -0.85 降到 ~0.28，远低于 0.50 阈值。让 progress 信号成为真正的正向主力，不再被惩罚抵消。保持 distance gate 不变（它已经在起作用），保持 landing proxy 不变（它是当前成功的关键）。

唯一改动：`w_vel: 0.01→0.003`, `w_angle: 0.01→0.003`, `w_angvel: 0.005→0.002`。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.003       # 【降低 3.3x】0.01→0.003，减轻速度惩罚对 progress 的抵消
    w_angle   = 0.003       # 【降低 3.3x】0.01→0.003，同上
    w_angvel  = 0.002       # 【降低 2.5x】0.005→0.002，同上
    w_landing = 0.03        # 保持不变：iter 5 降低后 crash，当前值已验证有效

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零

    # 距离门控：stability penalty 只在接近地面时生效
    D_gate = 3.0            # 距离阈值：超过此距离 stability penalty 被门控衰减
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty (DISTANCE-GATED) ---
    # 门控因子：距离远 → 0（自由机动），距离近 → 1（精细控制）
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    raw_penalty = w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel
    gate = max(0.0, 1.0 - dist_next / D_gate)  # ∈ [0, 1]，距离越近门越开

    stability_penalty = -raw_penalty * gate

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```