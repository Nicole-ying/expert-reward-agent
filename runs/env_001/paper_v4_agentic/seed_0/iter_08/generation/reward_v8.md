# 设计理由
第 0 步审计：所有观测维度均已使用，组件活跃率正常，agent 已成功着陆并达到目标分数。问题在于 `approach_bonus` 的尺度压倒性主导（98.8%），使 `progress_reward` 和 `landing_safety_penalty` 几乎沦为噪声，违背了奖励函数多目标平衡的设计意图。此为典型的 **尺度偏离**，而非信号缺失。

干预层级 **Level 1**（尺度修复）：
- **降低 approach_bonus 系数**：从 `2.0` 降到 `1.0`。原系数使该组件每步均值高达 `0.657`，掩盖其他信号。降为一半后预计每步约 `0.33`，仍为主信号，但为 progress 和 safety 留出梯度空间。
- **提升 landing_safety_penalty 系数**：将 `vel_pen` 权重从 `0.03` 提到 `0.10`，`ang_pen` 从 `0.02` 提到 `0.06`，`tilt_pen` 从 `0.03` 提到 `0.10`（约 3 倍增强）。因 `gate_safety` 门控在远处关闭，仅着陆附近激活，增强后不会过度惩罚探索；同时符合设计约束（估算惩罚 per-step 约 `0.002`，远小于主信号的 0.3x 警告线）。
- `progress_reward` 保持 `1.0` 不变——其绝对贡献虽小，但只代表高效路径上进度变化微小，属于正常现象，依靠 relative 比重提升即可。

该修改仅调节系数，不改变组件结构与数学形态，风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态（用于距离计算）
    x, y = obs[0], obs[1]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    l_contact = next_obs[6]
    r_contact = next_obs[7]

    # ---------- 1. 进度奖励：向目标 (0,0) 靠近 ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next
    progress_reward = 1.0 * progress

    # ---------- 2. 着陆预备与接触奖励 ----------
    prox = 1.0 / (1.0 + 10.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(nx_v) + abs(ny_v)))
    angle_factor = 1.0 / (1.0 + 3.0 * (abs(n_angle) + abs(n_ang_v)))

    # 双腿接触作为连续因子（二值乘积，0 或 1）
    contact_factor = l_contact * r_contact

    # 混合奖励：未接触时保留一半引导，接触时获得完整奖励
    approach_bonus = 1.0 * prox * speed_factor * angle_factor * (0.5 + 0.5 * contact_factor)

    # ---------- 3. 着陆安全性惩罚 ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate_safety = 1.0 / (1.0 + 5.0 * dist_next)
    landing_safety_penalty = (0.10 * vel_pen + 0.06 * ang_pen + 0.10 * tilt_pen) * gate_safety

    # ---------- 总奖励 ----------
    total_reward = progress_reward + approach_bonus - landing_safety_penalty

    components = {
        "progress_reward": float(progress_reward),
        "approach_bonus": float(approach_bonus),
        "landing_safety_penalty": float(landing_safety_penalty)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，僵尸组件无，终止模式为成功着陆；问题为 `approach_bonus` 尺度垄断，需重新平衡权重。
- **behavior**: agent 已学会稳定着陆（分数 >200），但梯度过分集中于 `approach_bonus`，其他组件形同虚设。
- **signal**: `approach_bonus` 强度约为 progress 的 300 倍，safety penalty 完全不可见。
- **level**: Level 1
- **hypothesis**: 降低 `approach_bonus` 主导性、适度强化安全惩罚，将使奖励函数各目标权重恢复平衡，agent 在保持成功率的同时获得更鲁棒的行为梯度。
- **risk**: 若 `approach_bonus` 下降过多，初期探索动力可能小幅减弱，但因其仍为主信号且安全惩罚在远处被门控消隐，收敛应保持稳定。