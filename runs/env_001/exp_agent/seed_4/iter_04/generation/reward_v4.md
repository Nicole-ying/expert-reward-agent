## 诊断

**Agent 状态**：episode 跑满 1000 步，无 crash，得分从 149.94 提升到 165.26——在进步，但仍有结构性问题。

**根因分析**：

| 组件 | mean | ratio_to_progress | nonzero_rate |
|------|------|-------------------|-------------|
| progress_delta_reward | 0.0025 | 1.0x | 99.97% |
| **soft_landing_proxy** | **0.0237** | **9.3x** | **99.16%** |
| stability_penalty | -0.0028 | 1.1x | 100% |

核心问题：`soft_landing_proxy` 仍然是 progress 的 9.3 倍（iter2 时是 117x，上次削到 0.03 后降到 9.3x）。它 99% 时间都在激活——说明三个 bounded 因子（proximity, smooth_speed, upright）几乎全程非零。

**为什么这是问题**：landing proxy 在 agent 远离目标时也在和 progress 竞争。理想情况下，远距离时 progress 主导导航，近距离时 landing proxy 主导着陆质量。现在两个信号全域混在一起，landing proxy 的量级一直压着 progress。

**上次改了什么**：把 `w_landing` 从 0.3 降到 0.03。有效果（ratio 从 ~117x 降到 9.3x），但还不够。继续削系数是层次 1，但 9.3x 仍然太高。这次改用**层次 2：加距离门控**——landing proxy 只在靠近目标时激活，从根本上解耦两个信号的作用域。

**修改方案**：给 landing proxy 加硬距离门控，dist > 1.5 时完全不触发，让 progress 在远距离独占主导权。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 软着陆连续代理奖励

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_gate = 1.5            # 【新增】距离门控：超过此距离 landing proxy 完全不激活
    D_max  = 2.0            # 距离阈值：超过此距离 proximity 归零
    V_max  = 3.0            # 速度阈值：超过此速度 smooth_speed 归零
    A_max  = 0.5            # 角度阈值（弧度，~28°）：超过此角度 upright 归零
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy (CONTINUOUS, DISTANCE-GATED) ---
    # 【修改】仅在 dist < D_gate 时激活，让 progress 在远距离独占主导
    if dist_next < D_gate:
        proximity    = max(0.0, 1.0 - dist_next / D_max)
        smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
        upright      = max(0.0, 1.0 - abs_angle / A_max)
        landing_bonus = w_landing * proximity * smooth_speed * upright
    else:
        landing_bonus = 0.0

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

**预期效果**：
- `soft_landing_proxy` 的 `nonzero_rate` 会从 99% 明显下降（取决于 agent 在 D_gate=1.5 内的时间比例）
- `ratio_to_progress` 会自然下降，因为远距离时 landing_bonus 贡献为 0
- agent 在远距离时只受 progress + stability 驱动，靠近目标后才叠加 landing quality 引导
- 如果下一轮得分停滞或倒退，说明门控太激进，可以放宽 D_gate 或改用软门控