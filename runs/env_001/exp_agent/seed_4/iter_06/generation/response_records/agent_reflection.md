# Response Record

## 诊断

### 1. Agent 发生了什么？
Episode 长度 76（远低于 150），60% 提前终止，得分 -54。**Agent 在 crash**——它根本没学会飞行，直接坠毁。

### 2. 哪个组件是主要原因？
看 ratio_to_progress：
- `stability_penalty`: **-0.874** — 惩罚几乎和 progress 奖励一样大！
- `soft_landing_proxy`: 0.171 — 太小，几乎没作用。
- `total_reward`: 0.297 — 每步净信号接近零，agent 没有有效的学习梯度。

**根因：w_landing 从 0.03 降到 0.01 是致命错误。** 对比历史记忆：
- iter 2-3（w_landing=0.03）：得分 150~165，episode 1000 步，landing proxy ratio 高达 100x+ — 它是实际的主学习信号。
- iter 5（w_landing=0.01）：landing proxy 萎缩到 ratio 0.17，progress 和 stability 几乎互相抵消，agent 失去引导→ crash。

**高 ratio 不是 bug，是 feature。** landing proxy 作为事件型奖励天然 ratio 偏高，但它是引导 agent 学会软着陆的核心信号。削弱它就等于关掉了导航。

### 3. 我之前改了什么？
上轮把 w_landing 从 0.03 降到 0.01，理由是"缓解 landing proxy 对 progress 的碾压"。但这是误判——那个"碾压"恰恰是成功的关键。本轮必须恢复。

### 修复策略

恢复 w_landing 到 0.03（历史最佳），同时做一个关键改进：**给 stability penalty 加距离门控**。当前 stability penalty 是全局生效的——agent 在高空下降时也被惩罚速度/角度，这会抑制必要的机动。距离门控让 penalty 只在接近地面（需要精细控制时）才生效，高空允许自由机动。

这是层次 2 改动（改数学形式）：只改 stability_penalty 一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 【恢复】0.01→0.03，回到历史最佳配置

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
    # 这样 agent 在高空下降时不会被速度/角度惩罚束缚
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

### 改动说明

| 改动 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `w_landing` | 0.01 | **0.03** | 恢复历史最佳配置。0.01 导致 landing proxy 失效，agent 失去导航信号→ crash |
| `stability_penalty` 形式 | 全局 `-raw` | **`-raw × gate`** | 新增距离门控。`gate = max(0, 1 - dist/D_gate)`，高空 gate≈0（自由机动），近地 gate→1（精细控制）。解决 iter 3-4 可能存在的"高空机动被过度惩罚→ plateau"问题 |

**为什么不是层次 1（只改系数）：** 单纯恢复 w_landing=0.03 只是回到 iter 3 的状态（165 分 plateau）。距离门控是让 stability penalty 更聪明——只在需要它的区域生效，这是 iter 3-4 未曾尝试过的改进方向。
