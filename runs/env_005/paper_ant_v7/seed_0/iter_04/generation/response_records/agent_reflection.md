# Response Record

# 设计理由
本轮不做结构性修改，仅进行尺度修复（Level 1）。训练反馈表明 `upright_penalty` 的每步平均量级（≈ ‑0.55）与主信号 `forward_reward` 的每步平均量级（≈ +0.62）几乎持平，导致步均 net 奖励为负，迫使 agent 选择“慢速保命”的保守策略。将 `upright_penalty` 的系数从 1.0 降至 0.3，使该惩罚步均负担降至约 ‑0.16，满足 ≤ 0.3 倍主信号的设计校准要求。同时保持 hinge 激活阈值 0.7 不变，确保危险倾向仍然受到约束。预期修改后步均 net 奖励转正，agent 会因正向总回报而敢于加速前进，同时 `height_reward` 继续提供生存动机。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 信号提取 ----------
    body_z = obs[0]
    quat_x = obs[2]
    quat_y = obs[3]
    body_x_velocity = obs[13]
    torque_penalty = sum([a**2 for a in action])

    # ---------- 身体直立投影 ----------
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---------- 高度安全门控（独立正向生存奖励） ----------
    low_factor = max(0.0, (body_z - 0.2) / 0.1)
    high_factor = max(0.0, (1.0 - body_z) / 0.1)
    height_factor = min(low_factor, high_factor)
    height_reward = 0.1 * height_factor

    # ---------- 主学习信号：前进速度 ----------
    forward_reward = 1.0 * body_x_velocity

    # ---------- 直立姿态约束（hinge 惩罚，系数降低） ----------
    upright_error = max(0.0, 0.7 - body_up_z)
    upright_penalty = -0.3 * upright_error   # 原系数 1.0 → 0.3

    # ---------- 力矩效率约束 ----------
    action_cost = -0.05 * torque_penalty

    # ---------- 总奖励 ----------
    total_reward = forward_reward + height_reward + upright_penalty + action_cost

    components = {
        "forward_reward": forward_reward,
        "height_reward": height_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 缺少对“接近终止高度”的直接惩罚，但 height_factor 已提供软化；核心问题是 upright_penalty 过重抑制了前进信号。
- **behavior**: agent 学会了长时间存活但移动缓慢，因为姿态惩罚使任何加速带来的倾斜代价过大，总奖励为负。
- **signal**: forward_reward 与 upright_penalty 的步均量级接近（0.62 vs 0.55），净回报为负，需降低惩罚权重。
- **level**: Level 1
- **hypothesis**: 将 upright_penalty 系数缩至 0.3 后，步均净回报转正，agent 在生存压力和高回报激励下会主动提高前进速度，同时保留姿态意识。
- **risk**: 若 agent 过度倾斜仍能获得正回报，可能导致频繁摔倒；但 height_reward 与高度终止提供反制，且 hinge 阈值仍保持 0.7，风险可控。
