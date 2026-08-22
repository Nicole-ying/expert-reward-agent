# Response Record

1. `evidence`：本轮score=-98.5、len=69.7，early terminal 17/20；contact_bonus episode_sum仅0.57且active_rate仅1.1%，orientation_penalty负向-3.3；历史best iter3使用相同代码获得91.8分，但当前run未复现，说明早期探索不足、学习信号不够密集。
2. `behavior_diagnosis`：策略尚未学会稳定下降，多数episode在早期就发生翻滚或crash，无法进入近地区域获得contact_bonus的正向引导，陷入早期失败。
3. `signal_completeness`：progress提供向目的地移动的奖励、orientation_penalty惩罚近地不良姿态、contact_bonus提供近地引导和接触奖励，职责完备；但contact_bonus的容忍度过严，导致早期学习信号过弱，可达性不足。
4. `selected_level`：Level 2，修改组件公式形式（在contact_bonus的soft_factor中引入training_progress调度，放宽早期姿态/速度容忍度）。
5. `selected_intervention`：基于best代码，修改contact_bonus的近地引导部分：将soft_factor中的方差参数乘以`scale = 1.0 + 4.0 * (1.0 - training_progress)`，使初期宽容、后期渐紧；接触奖励部分保留原本严格的固定参数。
6. `falsifiable_hypothesis`：若修改有效，首期训练中agent更容易因近地下降获得正奖励，从而快速踏入近地并学习安全姿态，最终习得成功着陆；外在指标应表现为score上升、ep_len增长、contact_bonus的episode_sum和active_rate增加。
7. `expected_next_round`：score应正向提升，len显著增长，contact_bonus的均值与active_rate上升，orientation_penalty小幅减小。
8. `main_risk`：若初期宽容过度，可能诱使agent在近地处摇摆而非稳定下降，后期收紧时已学策略难以适应，导致学习中断或崩溃。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact_bonus with curriculum ----------
    descent_target_vy = -0.5
    vy_error = vy_next - descent_target_vy

    # 调度因子：初期放大方差（容忍度更高），后期收缩至原值
    scale = 1.0 + 4.0 * (1.0 - training_progress)   # training_progress 0→1 ⇒ scale 5→1

    # 近地引导：使用宽松因子
    near_soft_factor = (2.718281828 ** (-(vy_error**2) / (0.1 * scale))) * \
                       (2.718281828 ** (-(vx_next**2) / (0.1 * scale))) * \
                       (2.718281828 ** (-(angle_next**2) / (0.05 * scale))) * \
                       (2.718281828 ** (-(angvel_next**2) / (0.05 * scale)))

    # 接触奖励：始终使用严格因子（原参数）
    contact_soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                          (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                          (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                          (2.718281828 ** (-(angvel_next**2) / 0.05))

    contact_bonus = 0.0
    if y_next < 1.0:
        contact_bonus = 5.0 * near_soft_factor   # 近地鼓励良好的下降与姿态（宽容）

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * contact_soft_factor # 接触时大额稳定性奖励（严格）

    # ---------- fuel penalty ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- near-ground orientation penalty ----------
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
