# Response Record

## 诊断

| 问题 | 证据 |
|------|------|
| **Agent 在 crash** | 100% early termination，episode length 仅 70.6，score = -109 |
| **稳定性惩罚碾压进度信号** | `stability_penalty` ratio_to_progress = **-9.07**，即惩罚力度是正向进度信号的 9 倍 |
| **软着陆引导形同虚设** | `soft_landing_proxy` nonzero_rate = 0.4%，几乎从不触发 |
| **净学习信号趋近于零** | `total_reward` mean = 0.016 ≈ 10×0.016(progress) + (-0.147)(penalty) → agent 怎么做都被惩罚抵消 |

**根因**：stability_penalty 太强且"无差别攻击"——agent 在远处需要大幅机动接近目标时，速度和倾斜就被重罚，导致它宁可 crash 也不敢动。

**本轮修改**：对 `stability_penalty` 引入**距离门控**（层次 2：改数学形式）。让稳定性约束只在接近目标时生效，远处允许自由机动。系数同步下调以匹配门控后的量级。只改这一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty with distance gating ==========
    # Rationale: stability matters most near the landing pad.
    # Gate = 1/(1+5*d) — near 0 when far, near 1 when close.
    # This lets the agent maneuver freely at distance, then stabilize on approach.
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)

    # Reduced base coefficients (roughly 3x lower), further attenuated by gate
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03

    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])

    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Soft landing proxy (task-completion hint) ==========
    # Unchanged from previous iteration — fix one thing at a time.
    near_target   = d_next < 0.2
    low_speed     = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    stable_angle  = abs(next_obs[4]) < 0.1
    both_contact  = (next_obs[6] == 1.0) and (next_obs[7] == 1.0)

    soft_landing_proxy = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # ========== Total reward ==========
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

### 改动说明

| 改动 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| 距离门控 | 无 | `1/(1+5·d_next)` | 远处 gate≈0（自由机动），近处 gate→1（施加约束） |
| `w_vel` | 0.1 | 0.03 | 配合门控下调，避免近处仍过强 |
| `w_angle` | 0.5 | 0.15 | 同上 |
| `w_omega` | 0.1 | 0.03 | 同上 |

**预期效果**：远处时 stability_penalty ≈ 0，progress_delta 成为唯一主导信号，agent 能自由学习接近目标；接近目标时门控打开，稳定性约束逐步生效，引导减速和调姿。如果下一轮 episode 显著变长（不再 crash），说明门控方向正确。
