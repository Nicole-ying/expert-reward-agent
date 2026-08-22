# 设计理由

经过信号覆盖审计发现：
- 几乎所有 episode（19/20）均为 truncation（超时），len ≈ 999，说明 agent 存活但不触发成功终止。
- `proximity_stability` 以静态绝对值形式发放奖励（基于 `next_dist`、`stab` 乘积、`contact_mult`），且在悬停姿态下 gate 底限 0.2 保障了持续非零输出。agent 由此学会在原点附近低速悬停并保持腿接触，以 88.6% 的 share 获取大量假奖励，而忽略 `progress_gated`。
- 核心信号缺口：缺少迫使 agent 真正完成“软着陆并终止 episode”的激励；绝对值奖励允许无限停留 exploit。

本轮处于正常模式，采取 **Level 2 结构变换**，将 `proximity_stability` 组件替换为 `landing_progress`，解决静态奖励 exploit：
1. **除去绝对值奖励**，改用 **距离改善量**（`max(0, dist - next_dist)`）作为移动过程引导，不再因“停在原点附近”获得分数。
2. **去掉 gate 底限**，让 `stability_gate` 在不稳定状态下直接趋零，避免不稳定步获得改善奖励。
3. **增加着陆完成推断**：当 `next_dist` 极小、速度低、角度小且至少一条腿接触时，推断为成功着陆，给予一次性的高奖励（相当于 terminal 硬信号，但完全从观测推导）。由于成功着陆后环境会立即终止 episode，不会存在持续 exploit。
4. 保留 `progress_gated`（向目标速度奖励）和 `fuel_penalty`（燃料节省）不变，维持多粒度引导。

数学形式：
- 若着陆条件满足：`reward = w_landing`
- 否则：`reward = w_delta * max(0, dist - next_dist) * stab_gate`，其中 `stab_gate = (1 - |n_angle|/0.5) * (1 - next_vel_abs/1.0) * (1 - |n_angvel|/2.0)`，无 clamp 最小值。
- 系数：`w_delta = 20.0`，预期 per‑step ≈ 0.3；`w_landing = 200.0`，与任务成功直接对齐。

校准检查：`fuel_penalty` per‑step ≤ 0.2，主信号合计约 0.66，惩罚/主信号约 0.3 < 0.5。gate 在“不理想但安全”区域（例如角度 0.3、速度 0.5、角速度 1.0）约为 0.42，高于 0.3，不会塌缩。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_delta = 20.0
    w_landing = 200.0
    w_fuel = 0.2

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    # ------------------- 1. velocity-toward-target progress signal -------------------
    dir_x = -nx / (next_dist + 1e-6)
    dir_y = -ny / (next_dist + 1e-6)
    toward_speed = nvx * dir_x + nvy * dir_y

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    gate = gate_angle * gate_vel * gate_angvel

    progress_gated = w_progress * max(0.0, toward_speed) * gate

    # ------------------- 2. landing progress (replaces proximity_stability) -------------------
    # Check plausible landing condition
    landing_cond = (next_dist < 0.15 and
                    (n_left + n_right) >= 1.0 and
                    abs(n_angle) < 0.3 and
                    next_vel_abs < 0.5 and
                    abs(n_angvel) < 0.5)

    if landing_cond:
        landing_progress = w_landing
    else:
        # stability gate without floor – zero if unstable
        a_stab  = max(0.0, 1.0 - abs(n_angle)  / th_angle)
        v_stab  = max(0.0, 1.0 - next_vel_abs   / th_vel)
        av_stab = max(0.0, 1.0 - abs(n_angvel)  / th_angvel)
        stab = a_stab * v_stab * av_stab

        distance_reduction = max(0.0, dist - next_dist)
        landing_progress = w_delta * distance_reduction * stab

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_gated + landing_progress + fuel_penalty

    components = {
        'progress_gated':   progress_gated,
        'landing_progress': landing_progress,
        'fuel_penalty':     fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 静态绝对值奖励 + 门控底限导致 exploit，所有观测已使用但缺少着陆完成激励。
- **behavior**: agent 在原点附近低速悬停、保持腿接触，最大化 `proximity_stability` 而避免触发终止，episode 超时。
- **signal**: `proximity_stability` 静态奖励过强，`progress_gated` 不足以驱策真正着陆。
- **level**: Level 2
- **hypothesis**: 将静态绝对值改为“距离改善量 + 一次性着陆奖励”，消除可持续获奖，使奖励与完成任务直接对齐。
- **risk**: 着陆条件若过于苛刻可能导致稀疏奖励；此时 `progress_gated` 和 `distance_reduction` 将继续提供密集引导，缓解该风险。