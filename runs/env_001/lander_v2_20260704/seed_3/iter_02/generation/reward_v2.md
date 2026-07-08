## 诊断与干预

`evidence`：所有20回合均在68.8步内以crash提前终止，score=-95.25；soft_landing_proxy的active_rate仅1.9%几乎无学习信号；stability_penalty与approach_reward每步均值约-0.018 vs +0.016，比值约1.09远超0.5预警线。

`behavior_diagnosis`：agent在到达目标平台前即全部crash——approach_reward虽在引导靠近，但同量级的全局stability_penalty惩罚一切位移和姿态变化，迫使agent在"前进获奖励"与"运动受惩罚"之间陷入冲突，无法学到有效的到达与着陆策略。

`signal_completeness`：approach_reward提供过程引导，stability_penalty提供姿态约束，soft_landing_proxy理论上提供着陆完成信号——职责基本完备，但soft_landing_proxy因乘积形态塌缩不可达，stability_penalty尺度异常压过主信号。

`selected_level`：Level 1——stability_penalty的数学形态（加权绝对值和）本身合理，证据主要指向其系数过强（|penalty/progress| ≈ 1.09 > 0.5），先做尺度修复。

`selected_intervention`：仅将stability_penalty三个系数全部降低10倍——w_speed: 0.01→0.001, w_angle: 0.1→0.01, w_angvel: 0.05→0.005，使|penalty/progress|目标值≈0.11，接近轻约束起点0.1。

`falsifiable_hypothesis`：降低约束权重后，approach_reward将主导早期学习，agent应能先学会到达目标平台附近，再逐步满足稳定性要求；若比例修复后agent仍全部crash，则说明问题不在相对尺度而在数学结构（转Level 2）。

`expected_next_round`：score应显著改善（更接近0），episode_length应增长（agent不再快速crash），terminated中应有部分成功着陆而非全部crash，stability_penalty的signed_share应降至10%以下，approach_reward应成为主导正信号。

`main_risk`：约束过弱可能导致agent以高速/大倾角砸向平台，虽到达目标但因冲击过大触发crash——此时需在下一轮考虑将stability_penalty改为距离门控（仅在平台附近生效）而非全局作用。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_support_contact, 7: right_support_contact
    # goal is at (0,0).

    # 1. Main learning signal: progress toward the landing platform.
    d_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    d_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    approach_reward = d_prev - d_next

    # 2. Stability constraint: light penalty on large speeds, tilt, and angular velocity.
    speed = abs(next_obs[2]) + abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    w_speed = 0.001
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = -w_speed * speed - w_angle * angle - w_angvel * ang_vel

    # 3. Soft landing proxy: boost when almost on target, slow, upright, and both feet contact.
    dist = d_next
    contact_both = next_obs[6] * next_obs[7]
    k_dist = 2.0
    k_speed = 1.0
    k_angle = 5.0
    w_proxy = 5.0
    soft_landing_proxy = (w_proxy * contact_both *
                          (2.718281828 ** (-k_dist * dist)) *
                          (2.718281828 ** (-k_speed * speed)) *
                          (2.718281828 ** (-k_angle * angle)))

    total_reward = approach_reward + stability_penalty + soft_landing_proxy

    components = {
        "approach_reward": approach_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```