# 诊断报告

## evidence
- 记忆表：`iter_1` score=233.56（best，len=1092）；`iter_2` 加入 energy_penalty 后降到 187.89；`iter_3` 同时把 forward 去饱和（weight=3, scale=3）并把 angle_weight 提到 5.0，结果崩到 -55.52（len=292，17/20 early terminal）。
- `iter_1` 最终组件：`forward_progress_reward` ep_sum=412.14，magnitude_share=98.5%；`upright_stability_penalty` ep_sum=-6.07，share=1.5%。奖励几乎由单一前进速度信号垄断。
- `iter_1` checkpoint：forward 从 cp50% 的 71 单调涨到 cp100% 的 426，而 stability 全程只有 -0.2 ~ -2.5。前进项在正常驱动学习，稳定项几乎不起作用。
- 关键换算：`score / forward_progress_reward` 在 iter_1=0.567、iter_2=0.566，几乎恒定 → 外部得分本质上就是「前进距离/速度积分」的缩放。要上 300，需要更多前进积分。
- `iter_1` len=1092、terminated=20/20 但 score=233<300 → 回合是**摔倒终止**而非到达地形终点，摔倒限制了最终距离。
- 隐患：`iter_1` 的 forward 项只依赖 `horizontal_velocity`，前向摔倒时水平速度仍为正，因此 agent **在摔倒过程中还在拿前进奖励**，等于奖励了「前扑摔倒」这个失败模式。

## behavior_diagnosis
当前策略学到的是一条能存活 1092 步、速度约 0.4 的步态，但最终仍以摔倒结束。因为前进奖励与姿态完全解耦，agent 没有足够动机在推进的同时保持躯干直立；它可以在倾斜甚至前扑时继续累积前进奖励。这解释了为什么外部得分停在 233（<300）：不是走不快，而是最终摔倒把距离截断。该模式不健康。

## signal_completeness
- `forward_progress`：存在，且是唯一主驱动。
- `upright_stability`：存在但极弱（share 1.5%），且 iter_3 用 5x 二次惩罚加强它造成悬崖式奖励、训练崩溃，不能简单复刻。
- **缺失信号**：前进奖励缺少「姿态门控」——应该只在躯干接近直立时才全额给前进奖励，倾斜/摔倒时不给。这正是环境卡 `failure_modes` 建议的「将 forward_progress 与 upright_stability 联合考虑；给前进速度添加要求躯干接近直立的门限」。

## selected_level
**L2**（改 forward_progress 组件的数学形态）。best=233.56 距目标 300 差距约 22%（>10%），不是纯 L1 微调场景；但 iter_1 核心已被证明有效，所以保留其 base 前进项与稳定项，只给前进项加一个姿态门控，不做框架级重写。

## selected_intervention
目标组件：`forward_progress_reward`。
- 保留 iter_1 的归一化速度 base：`v / sqrt(v^2 + 1)`。
- 乘以直立门控：`gate = exp(-max(0, |avg_angle| - 0.5) / 1.0)`。
- 死区 0.5 rad 内 gate=1（正常行走几乎不受影响），倾斜越大 gate 越接近 0，前向摔倒时不再累积前进奖励。
- `upright_stability_penalty` 保持 iter_1 的轻量二次形式不变，避免 iter_3 的悬崖问题。

## falsifiable_hypothesis
若 iter_1 的终端摔倒主要来自「前扑时仍被前进项奖励」，那么门控会移除摔倒过程中的正奖励，使 agent 学会「在保持直立的同时前进」。预计摔倒率下降、前进积分上升、score 向 300 靠近。若下一轮速度明显下降（agent 变得过度保守）或摔倒依旧，则需要回头调 gate 的温度 tau 或轻微调整稳定项。

## expected_next_round
策略应保持约 0.4 的前进速度（死区内不受门控影响），但在倾斜阶段减少前扑式加速；len 维持或略升，forward_progress_reward 的 ep_sum 因门控略降，但外部分数应更接近 250–300，摔倒导致的低分尾部减少。

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
    angle_deadband = 0.5
    gate_tau = 1.0
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