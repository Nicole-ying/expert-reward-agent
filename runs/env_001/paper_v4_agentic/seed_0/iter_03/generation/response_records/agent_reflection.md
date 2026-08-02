# Response Record

# 设计理由
本轮正常模式，只修改 **landing_contact_bonus** 组件。
- **根因**：上一轮训练反馈显示 `landing_contact_bonus` 的量级（episode_sum_mean≈365）压倒性统治总奖赏，而 `progress_reward` 每步平均仅 0.003，agent 学会了在远端踩地获取接触奖励（active_rate 72.3%），长期存活（len≈872）且极少完成着陆（terminated=3/20），累积迭代记录中 len 从 211 暴涨到 872，说明 exploit 形成了。
- **变换**：将门控 `gate = 1/(1+5*dist)` 改为更尖锐的 `sharper_gate = gate²`，同时将接触奖励系数从 0.3 降至 0.1。使得当距离较远时（dist=1，原 gate≈0.167，平方后≈0.028，每步奖励从 0.05 降至 0.003），与 progress 量级相当；接近目标时（dist=0.2，原 gate=0.5，平方后 0.25，0.1×0.5≈0.05）仍能给出有效的着陆姿态信号。从而迫使 agent 向中心靠近才能获得显著正反馈。
- **不变**：`progress_reward`、`landing_safety_penalty`、`x_boundary_penalty` 保留原样（后者虽 dead 但本轮不动）。
- **系数校准**：新接触奖励在目标附近最大 per-step ≈ 0.1×2×1=0.2，主信号 progress per-step ≈ 0.003，倍率约 66 倍，但由于触发范围极窄且成功着陆后 episode 终止，不会累积过重；在远处接触奖励几乎消失，不会淹没 progress。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next          # positive when getting closer
    progress_reward = 1.0 * progress

    # ---------- 2. Horizontal boundary penalty (unchanged, noted dead) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty (unchanged) ----------
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)

    gate = 1.0 / (1.0 + 5.0 * dist_next)     # proximity gate
    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- 4. Landing contact bonus: sharper gate & reduced coefficient ----------
    sharper_gate = (1.0 / (1.0 + 5.0 * dist_next)) ** 2.0
    landing_contact_bonus = 0.1 * (n_left + n_right) * sharper_gate   # coeff 0.3->0.1

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty + landing_contact_bonus

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty),
        "landing_contact_bonus": float(landing_contact_bonus)
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测维度均已使用，但 `landing_contact_bonus` 的门控过于宽松，允许 agent 在远离目标处踩地获取高回报，导致 exploit。
- **behavior**: agent 学会了在半途用双腿触地以获取大额奖励，长期存活但不完成着陆，terminated 率极低（3/20）。
- **signal**: 主前向奖励（progress）被接触奖励淹没；远端接触奖励过强，目标附近信号被稀释。
- **level**: Level 2（结构变换：门控尖锐化+系数降低）
- **hypothesis**: 尖锐门控与降低系数将迫使 agent 必须驶向目标附近才能获得有效的接触奖励，进而恢复 progress 的主导地位，推动 agent 完成着陆。
- **risk**: 若接触信号过弱，可能削弱姿态调整动力；但安全惩罚与 progress 仍在，预期 agent 仍能学会减速靠近，下一步可依反馈进一步调整。
