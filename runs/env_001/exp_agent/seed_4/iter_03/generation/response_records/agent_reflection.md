# Response Record

## 诊断

### 1. Agent 发生了什么？

从 iter1 到 iter2 是巨大飞跃：从 crash（len=71, score=-107）到完整 episode（len=1000, score=150）。但现在的问题是 **landing proxy 劫持了整个学习信号**。

### 2. 哪个组件是主因？

| 组件 | mean | ratio_to_progress |
|------|------|-------------------|
| progress_delta_reward | 0.00185 | 1.0x |
| soft_landing_proxy | **0.234** | **126x** |
| stability_penalty | -0.00245 | 1.3x |

`soft_landing_proxy` 是 progress 信号的 **126 倍**。agent 每步从 landing proxy 获得 ~0.23 奖励，而从实际前进获得 ~0.002。它学会的策略是：靠近降落点、保持慢速、保持竖直——然后**待在那里吃 bonus**，而不是高效完成着陆。progress_delta 极小（0.002/步 × 1000步 ≈ 总共只靠近了 2 单位距离），说明 agent 基本在 hover。

### 3. 上轮改了什么？

Iter1→2：把 landing proxy 从稀疏二值改成连续 bounded 乘积（`max(0,1-x/D)`），并把 w_landing 从隐含的小值提升到 0.3。效果：不再 crash，但 proxy 过于强大。

### 修复方案

**层次 1 — 改系数**：大幅削弱 `w_landing`，从 0.3 降到 **0.03**（10x 削减）。这样 landing proxy mean 预计从 0.234 降到 ~0.023，ratio 从 126x 降到 ~12x。仍然提供靠近目标区域的引导，但不再淹没 progress 信号。

只改一个系数，下一轮可归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 软着陆连续代理奖励（从 0.3 大幅降低，避免劫持 progress）

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零
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
