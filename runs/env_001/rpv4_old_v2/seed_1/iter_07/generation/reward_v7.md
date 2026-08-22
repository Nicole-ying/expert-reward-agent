1. `evidence`：当前score=-20.44，contact_bonus的active_rate暴跌至13.7%但ep_sum_mean=3244.9（magnitude_share=98.9%），说明只有极少数回合发生了有效接触并获得了极大奖励，导致策略极不稳定；历史最佳iter3的score=91.82，其contact_bonus_active=94.4%、ep_sum_mean=1029，说明简单、带宽较宽的接触奖励能稳定驱动着陆行为。
2. `behavior_diagnosis`：agent几乎丧失了稳定着陆能力，大部分回合因坠毁或超时失败，仅少数侥幸回合获得极高额回报，导致整体得分仍为负值。
3. `signal_completeness`：当前奖励结构包含过程引导（progress）、软着陆奖励（contact_bonus）、燃料惩罚和近地姿态惩罚，职责完备；但contact_bonus的数学形式过于苛刻（极窄的指数窗口），使得本应是主成功信号的着陆奖励变得不可达，引导作用丧失。
4. `selected_level`：Level 2，因为contact_bonus的数学形态传达了错误的激励（极端稀疏），需要改变公式使其变得平滑且更容易触发。
5. `selected_intervention`：唯一目标组件为contact_bonus，将其从“近地连续引导 + 极苛刻接触奖励”改为“只在双腿触地时给予带宽更宽的稳定性奖励”，把衰减系数从`(-5.0*speed_sq)`、`(-10.0*angle²)`、`(-5.0*angvel²)`放宽为`(-1.0*speed_sq)`、`(-2.0*angle²)`、`(-1.0*angvel²)`，并将基础系数提升至100.0。
6. `falsifiable_hypothesis`：放宽稳定性条件后，着陆奖励将更容易被触发，agent会重新学会以较低但可接受的速度和姿态着陆，从而显著提高terminated率与平均得分。
7. `expected_next_round`：score应转正并有望接近历史最佳91.8或更高，terminated回合数应明显增加（目标≥15/20），contact_bonus_active_rate应回升至80%以上，contact_bonus的ep_sum_mean会大幅下降但保持正向。
8. `main_risk`：过度放宽条件可能导致agent学会以次优姿态（不完全竖直、速度略高）着陆，虽然能完成episode但后续可能难以进一步优化到更高分数；需在下一轮通过诊断确认是否需要收紧条件或增强progress来提升质量。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward (kept from best) ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact bonus: simpler, more generous stability reward ----------
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        stability = (2.718281828 ** (-1.0 * speed_sq)) * \
                    (2.718281828 ** (-2.0 * (angle_next**2))) * \
                    (2.718281828 ** (-1.0 * (abs(angvel_next)**2)))
        contact_bonus = 100.0 * stability

    # ---------- fuel penalty (kept unchanged) ----------
    fuel_penalty = -0.05 if action != 0 else 0.0

    # ---------- near-ground orientation penalty (kept unchanged) ----------
    orientation_penalty = 0.0
    if y_next < 0.5:
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```