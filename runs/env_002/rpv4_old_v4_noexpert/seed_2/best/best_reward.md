## 1. 诊断报告

**evidence**
- 记忆表：iter5 `score=284.15`（new best），`len=1026.9`，`terminated=20/20`；此前 iter1=233.56、iter4=190.75。
- iter5 最终组件：`forward_progress_reward` ep_sum=`437.03`、active=100%、magnitude_share≈100%；`upright_stability_penalty` ep_sum=`-0.132`、active=63.7%、magnitude_share≈0%。
- score_range=`[9.02, 300.60]`：单集最高已到 300.6（达标），但存在一个 9.02 的早期摔倒低分集，显著拉低均值。
- checkpoint 趋势：score 从 cp80 后稳定在 ~288（cp80=277.1, cp90=288.6, cp100=288.2），forward 稳定在 ~445，无骤升/骤降，说明 forward 信号没有被 exploit。
- iter4→iter5 改善原因：iter5 把 `gate_tau` 从 1.0 放宽到 2.0（L1），forward ep_sum 369→437，upright penalty -0.273→-0.132，score 190.75→284.15；证据明确，是 gate 放宽让快速步态得以保留，同时仍挡住大幅倾斜/前扑。

**behavior_diagnosis**
策略已经学会稳定且较快的双足前进步态，主体行为是健康的；训练后期分数平台化而非下滑，说明没有明显 reward hacking。当前主要短板是**鲁棒性尾端**：个别 seed 上发生早期摔倒（min score=9.02），把均值从 ~298 拉低到 284。upright_stability_penalty 目前 ep_sum 只有 -0.132、量级占比 0.0%，几乎是一个惰性信号，无法在摔倒前给 agent 足够强的“危险姿态/角速度”负反馈。

**signal_completeness**
- `forward_progress` 职责完备且已证明有效，保持不变。
- `upright_stability` 职责**名存实亡**：gate 通过乘性方式抑制了倾斜时的 forward 收益，但独立的稳定性惩罚量级过小，缺乏对“接近摔倒”状态的明确局部负反馈。
- 能量惩罚、地形 LIDAR、success/failure 信号按环境卡和过往经验不应引入。

**selected_level**
L1。当前 `284.15` 距目标 `300` 差距 5.3%（<10%），且 iter5 刚由 gate 放宽带来 +93 的改善，框架已被验证。只对稳定性惩罚做系数微调，不改数学形态、不动 forward 主信号。

**selected_intervention**
目标组件 `upright_stability_penalty`：
- `angle_weight`: 1.0 → **1.5**
- `angvel_weight`: 0.05 → **0.10**

保留 `angle_deadband=0.5`、`angvel_deadband=1.5`、二次死区形式和 `gate_tau=2.0` 不变。这样只增强死区之外的大倾斜/大角速度惩罚，不触碰正常步态。

**falsifiable_hypothesis**
早期摔倒尾端之所以存在，是因为当前稳定性惩罚量级≈0，agent 在接近摔倒状态时几乎没有独立负反馈可学习规避。适度增强该惩罚会提供非零但远小于 forward 的稳定性梯度，预期减少 catastrophic early-fall，使均值向 300 靠拢，同时不显著降低平均前进速度（forward 仍主导）。

**expected_next_round**
预期 score 均值上升（目标 ≥300），最低分提高，`len` 相近或略增；`forward_progress_reward` 仍 ~430–445 且 magnitude_share≈100%；`upright_stability_penalty` ep_sum 小幅增大（预计 -0.2 ~ -0.8 量级）。若出现 forward 明显下降、步态变僵硬，说明惩罚过强，下一轮回调系数。

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

    # --- upright stability (light deadbanded penalty) ---
    # iter5 showed the penalty is nearly inert (ep_sum=-0.132, share~0.0%),
    # leaving rare early-fall episodes under-penalized. Modestly strengthen
    # both angle and angular-velocity penalties so approaching-fall states get
    # a real but still small negative signal, without dominating forward.
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