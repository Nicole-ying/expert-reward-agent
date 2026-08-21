## 1. 诊断报告

**evidence**
- 记忆表：`iter_6` score=`306.15`，已超过目标 300；`len=1084.5`；`terminated=20/20`、`truncated=0/20`；`score_range=[303.9, 308.7]` 极窄，0/20 early-terminal。
- `iter_5→iter_6` 提升 `+22.0`（未超过 30 分阈值），上一轮改动为 L1：`angle_weight 1.0→1.5`、`angvel_weight 0.05→0.10`。
- 最终组件：`forward_progress_reward` ep_sum=`444.80`、active=`100%`、magnitude_share=`98.5%`；`upright_stability_penalty` ep_sum=`-6.86`、active=`93.8%`、share=`1.5%`。
- checkpoint 趋势：score 从 cp10 `-116.2` 爬升到 cp100 `306.4`，无中后期崩溃；`forward` 从 `-7.6` 增长到 `445`，无 exploit 式单调爆炸；`upright` 小且稳定。
- 历史教训：`iter_2` 加 energy_penalty 回退到 `187.89`；`iter_3` 大改崩到 `-55.52`；`iter_4` gate 过紧 `190.75`；`iter_5` gate_tau=2.0 + 轻 penalty 才新 best。

**behavior_diagnosis**
- 策略已学到稳定前进步态：20 集评估全部在约 1084 步终止，分数区间极窄，无早期摔倒、无 proxy 徘徊。`forward` 占主导符合本任务“前进为主”的定位；`upright` 小而活跃，起兜底防摔作用。行为健康，目标已解决，无需任何结构性改动。

**signal_completeness**
- 主职责齐备：`forward_progress`（obs[2]）有界归一化前进信号；`upright_stability`（obs[0]、obs[1]）带死区惩罚。能量效率为次级目标，且 `iter_2` 已验证加入会伤害主目标，保持不引入；lidar/contact 无必要。信号完备。

**selected_level**：L1
- 理由：当前 306 已超目标 300（差距仅 2%），按规则接近目标时必须避免 L2/L3；历史多次证明接近目标时大改会回退。仅做系数微调以巩固并预留安全边际。

**selected_intervention**
- 目标组件：`forward_progress_reward`。
- 具体修改：仅 `forward_weight` 从 `1.0` 提高到 `1.1`。其余数学形态、upright gate、stability penalty 全部保持不变。

**falsifiable_hypothesis**
- 小幅强化已验证的前进主信号，预计在保持 upright gate/penalty 不变的前提下轻微提升平均前进速度，使分数稳定在 300 以上并增加少量余量。若下一轮早期摔倒增多或分数跌破 300，说明前进信号过强压制了稳定性，应回退 `forward_weight=1.0`。

**expected_next_round**
- score 预计 `305~315`，len 仍约 1080（速度略增时可能略短），`terminated=20/20`，score_range 仍窄；`forward` share 略升至 ~98.6%，`upright` share 略降至 ~1.4%，无死组件。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Locomotion reward: posture-gated forward progress + upright stability.
    # Only uses environment_card-declared obs/next_obs indices.

    # --- forward progress base (proven iter_1 core) ---
    # iter_6 already solved target (306.15 > 300) with a tight score range.
    # L1: modestly raise forward_weight 1.0 -> 1.1 to add a small safety
    # margin above the target, without touching the stability mechanism that
    # removed the rare early-fall episodes in iter_6.
    forward_weight = 1.1
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
    gate_tau = 2.0
    avg_angle = 0.5 * (obs[0] + next_obs[0])
    gate_angle_excess = abs(avg_angle) - angle_deadband
    if gate_angle_excess < 0.0:
        gate_angle_excess = 0.0
    upright_gate = 2.718281828 ** (-gate_angle_excess / gate_tau)

    forward_progress_reward = base_forward * upright_gate

    # --- upright stability (unchanged deadbanded penalty from iter_6) ---
    # Keep the strengthened angle/angvel penalties that removed early falls,
    # leaving forward progress as the sole adjusted component this round.
    angvel_deadband = 1.5
    angle_weight = 1.5
    angvel_weight = 0.10
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