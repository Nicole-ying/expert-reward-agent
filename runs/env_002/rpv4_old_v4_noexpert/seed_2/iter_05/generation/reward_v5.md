## 1. 诊断报告

**evidence**
- 记忆表：iter1 best=`233.56`（len=1092），iter2=`187.89`，iter3 崩溃 `-55.52`，iter4=`190.75`（len=1053）。
- iter4 最终组件：`forward_progress_reward` ep_sum=`369.462`、magnitude_share=99.9%；`upright_stability_penalty` ep_sum=`-0.273`、active_rate=80.4%。
- 对比 iter1：forward ep_sum `412.142`，upright penalty `-6.070`，len `1092`。iter4 的 forward 奖励比 iter1 少约 `10%`，score 也低约 `42` 分。
- iter4 checkpoint：score 从 cp70% `135.4` 升到 cp100% `244.2`，forward ep_sum 最终 `420.573`；但最终固定评估 score 只有 `190.75`，说明策略方差较大、门控对速度的抑制作用在不同种子下明显。
- iter4 代码中 `upright_gate` 的 `gate_tau=1.0`、`angle_deadband=0.5`。gate 导致 upright penalty 几乎为 0（较 iter1 的 -6.07 大幅下降），说明策略学会了保持直立；但同时 forward 收益被明显压低，说明 gate 过严，自然步态的前倾—速度收益被过度抑制。

**behavior_diagnosis**
- 当前策略不是不健康，也没有 proxy 徘徊：它能稳定前进到 len≈1053，最终 20/20 摔倒终止说明其稳定步态仍会在长距离后失效，但整体是“走得远但不快”。
- iter4 的 gate 使策略过度强调直立，牺牲了前进速度；forward ep_sum 比 iter1 低 10%，这是 score 没有恢复 best 的主要候选因素。
- 当前得分距离目标 300 还有约 37% 差距，需要提升 forward reward 积分而不回到 iter3 的暴力和不稳定。

**signal_completeness**
- 必要职责基本完备：forward（obs[2]）、upright（obs[0]/obs[1]）都在使用。缺少的不是新信号，而是 forward 与 upright 之间的平衡：gate 偏严导致前进潜力受抑制。
- 不需要引入 LIDAR、接触相位或能量罚项，以免增加不稳定因素。

**selected_level**
- **L1**：只调 `gate_tau` 系数，不改数学表达式。
- 理由：iter4 的 gate 方向正确（移除了前扑漏洞、稳定了步态），但力度过紧。这属于“方向对但力度不对”的典型场景，适合 L1 微调，避免大改破坏已有积极结果。

**selected_intervention**
- 目标组件：`forward_progress_reward` 中的 `upright_gate`。
- 具体修改：将 `gate_tau` 从 `1.0` 提升到 `2.0`，其他参数保持不变。
- 这使 gate 对中等倾斜角的惩罚衰减更慢：在倾角略超过 deadband 时，forward reward 保留更多，从而释放自然步态的速度潜力，同时仍然保留对大幅倾斜的强约束。

**falsifiable_hypothesis**
- 放宽 gate 后，策略在速度—姿态权衡中会向更快步态偏移，forward ep_sum 预计回升至 iter1 水平（约 400~420），并可能进一步超过 iter1，因为 gate 仍能防止纯粹前扑式前进。
- 同时 upright penalty 可能小幅上升，但不会像 iter3 那样剧烈，因为 gate 依然存在且 angle_weight 仍为 1.0。
- 如果本轮 score 仍未接近或超过 best，则说明 gate 不是主要瓶颈，下一轮应重新考虑 forward 饱和结构。

**expected_next_round**
- 预期策略会在保持稳定（不频繁早期摔倒）的前提下提高平均前进速度，forward ep_sum 回升，len 可能接近或略低于 iter1，score 有望接近或超过 `233.56` 的 best。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Locomotion reward: posture-gated forward progress + upright stability.
    # Only uses environment_card-declared obs/next_obs indices.

    # --- forward progress base (proven iter_1 core) ---
    forward_weight = 1.0
    speed_scale = 1.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    base_forward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright gate on forward progress ---
    # Forward reward is paid in full only while the torso is near upright.
    # During a forward fall horizontal velocity stays positive, so the old
    # reward kept paying the agent while it was falling forward. This gate
    # removes that perverse reward without touching the proven speed base.
    #
    # iter4 used gate_tau=1.0, which suppressed natural gait speed too much.
    # Relax gate_tau to 2.0: gate decays more slowly for moderate tilt,
    # allowing faster gait while still blocking large tilt/fall exploitation.
    angle_deadband = 0.5
    gate_tau = 2.0
    avg_angle = 0.5 * (obs[0] + next_obs[0])
    gate_angle_excess = abs(avg_angle) - angle_deadband
    if gate_angle_excess < 0.0:
        gate_angle_excess = 0.0
    upright_gate = 2.718281828 ** (-gate_angle_excess / gate_tau)

    forward_progress_reward = base_forward * upright_gate

    # --- upright stability (unchanged light deadbanded penalty) ---
    angvel_deadband = 1.5
    angle_weight = 1.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    total_reward = forward_progress_reward + upright_stability_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty)
    }

    return float(total_reward), components
```