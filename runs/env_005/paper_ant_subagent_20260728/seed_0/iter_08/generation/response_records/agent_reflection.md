# Response Record

# 设计理由

本轮我选择**不延续 upright-gated forward + upright_bonus 的骨架**，而是**取消对前向速度的直立门控**，改用**高度门控直接乘前向速度**，并将直立的奖励改为**阈值惩罚**。  
这从根源上改变了 upright 与 forward 的耦合方式：之前“必须直立才能获得前进收益”会让 agent 在直立难维持时直接放弃前进（exploit 跌倒前的低收益区），而新设计让前向收益随时可得，仅用惩罚抑制明显摔倒，从而解放探索。

**数学形式与系数校准**  
- `forward`：`v_x * height_gate`，系数 1.0，在当前 len 约 986 时 per‑step 约 1.1 – 1.2。  
- `height_gate`：在安全区 (0.35 ≤ z ≤ 0.85) 为 1，渐变过渡区保证每步有梯度；完全在终止边界的 60 % 处开始衰减。  
- `upright_penalty`：hinge `max(0, threshold - up_z)`，阈值 0.7，系数 0.5（正常站立时 penalty=0，仅倾斜时触发，最大单步惩罚幅度约 0.5 – 0.8，远小于主 forward 信号）。  
- 其余惩罚保持温和：`lateral_penalty` 二次，系数 0.3；`action_penalty` 系数 0.005。  
- 总惩罚步均 < 0.5 × 主信号，满足设计校准。

**与历史版本的差异**  
- 无 `upright_bonus` 独立正向奖励；  
- 无 `upright_gate` 乘以 `forward`；  
- 首次使用 `height_gate` 直接门控 `forward` 并结合 `upright_hinge_penalty`。  
此组合从未出现在任何历史奖励中，且结构上避免了被拒草案（upright_bonus + height_gate）的核心模式。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- height gate: 1.0 in safe zone, decays to 0 near hard limits ----
    z_low  = 0.25   # termination boundary: z <= 0.2
    z_high = 0.95   # termination boundary: z >= 1.0
    z_safe_low  = 0.35
    z_safe_high = 0.85

    low_factor = (body_z - z_low) / (z_safe_low - z_low)
    low_factor = max(0.0, min(1.0, low_factor))

    high_factor = (z_high - body_z) / (z_high - z_safe_high)
    high_factor = max(0.0, min(1.0, high_factor))

    height_gate = low_factor * high_factor

    # ---- forward progress (main signal, gated only by height) ----
    w_fwd = 1.0
    forward_reward = w_fwd * v_x * height_gate

    # ---- upright penalty (hinge: only active when up_z < threshold) ----
    upright_threshold = 0.7
    upright_deficit = max(0.0, upright_threshold - up_z)
    w_up_pen = 0.5
    upright_penalty = -w_up_pen * upright_deficit

    # ---- lateral stability (quadratic penalty) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothness ----
    w_act = 0.005
    action_penalty = -w_act * sum(a ** 2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = forward_reward + upright_penalty + lateral_penalty + action_penalty

    components = {
        "forward_reward":   forward_reward,
        "upright_penalty":  upright_penalty,
        "lateral_penalty":  lateral_penalty,
        "action_penalty":   action_penalty,
        "_height_gate":     height_gate
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 终止前兆①高度越界无软 gate（height_penalty 几乎不触发），②直立信号过强且与 forward 耦合导致 forward 收入被压制——当前结构漏掉了高度安全连续门控，且过度限制了 forward 探索。
- **behavior**: agent 维持高直立度（up_z≈0.93）以获取 upright_bonus 并避免 gate 削减，但 forward 速度仅 ~1.16，无法突破到高分。
- **signal**: 缺独立的高度安全软门控；upright 过度参与正向奖励与门控，导致 agent 保守。
- **level**: Level 2
- **hypothesis**: 解除直立对前向的门控，让前向直接获高梯度；用 hinge 惩罚抑制明显摔倒，避免 agent 为赚取 upright_bonus 而放弃加速。这样 exploration 会偏向快速移动同时仍维持安全高度与大致直立。
- **risk**: 若 hinge 惩罚太温和，可能诱发大倾斜导致摔倒，但阈值 0.7 和系数 0.5 在多数训练阶段会提供足够约束，且高度 gate 能提前衰减奖励防止灾难性越界。
