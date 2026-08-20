`evidence`：最终策略20轮全部truncated（1000步），0次terminated，descent_quality占据72.5%奖励份额（episode_sum_mean≈1662），contact_quality激活率0%，velocity_penalty和attitude_penalty几乎为零。agent在y≈0附近低速、小角度持续获取descent_quality奖励，但从不触地。

`behavior_diagnosis`：agent陷入了经典proxy exploitation——在平台表面上方悬停，靠"接近地面+低速+姿态稳定"这一持续状态奖励维持高额累计收益，但从未完成双腿触地的任务终止条件。

`signal_completeness`：proximity提供趋近引导，descent_quality本应引导软着陆质量，但其数学形态（纯状态值）允许agent不触地即可获取高奖励。缺失的职责是"必须触地才能解锁着陆质量奖励"这一对齐约束。

`selected_level`：Level 2——证据直接否定当前descent_quality的纯状态值形态（state_to_improvement/proxy_to_completion_alignment模式匹配：agent占据好状态即可持续获奖，无需完成任务）。

`selected_intervention`：对descent_quality施加contact_gate（软门控），使其在无接触时仅保留5%基准值，有接触时获得完整值。接触由left_contact+right_contact>0.5判定。这是proxy_to_completion_alignment变换——将着陆质量代理信号对齐到任务完成条件（触地）。

`falsifiable_hypothesis`：悬停时descent_quality应下降约20倍，迫使agent为获取完整奖励而触地；一旦触地被发现，agent应增加接触频率，最终导向双腿稳定着陆。

`expected_next_round`：descent_quality的episode_sum_mean应大幅下降（无接触时约为当前的5%），contact_quality的active_rate应从0%上升，terminated比例应从0/20上升（出现成功或失败终止），episode平均长度可能先波动后下降。

`main_risk`：若agent长期无法发现触地动作，5%的残留descent_quality加proximity可能不足以维持稳定探索，导致策略退化或硬撞击地面。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from observations
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Proximity to target (assumed at origin)
    dist = (x**2 + y**2) ** 0.5
    proximity = 1.0 / (1.0 + dist)          # bounded [0,1]
    w_proximity = 1.0
    comp_prox = w_proximity * proximity

    # Shared stability factors reused across components
    speed_norm = (vx**2 + vy**2) ** 0.5
    factor_vel = 1.0 / (1.0 + speed_norm)       # bounded [0,1], 1 when still
    factor_angle = 1.0 / (1.0 + abs(angle) + abs(angular_vel))  # bounded [0,1]

    # Descent quality: now contact-gated to prevent hovering exploitation.
    # Without contact, only 5% of the quality is granted, creating a ~20x
    # incentive to touch down. With any contact, full quality is awarded.
    height_factor = 1.0 / (1.0 + abs(y))        # peaks at y=0
    contact_sum = left_contact + right_contact  # 0, 1, or 2
    contact_gate = 0.05 + 0.95 * min(1.0, contact_sum)  # 0.05 when no contact, 1.0 with contact
    descent_quality = contact_gate * height_factor * factor_vel * factor_angle
    w_descent = 3.0
    comp_descent = w_descent * descent_quality

    # Sustained contact quality: rewards stable, settled contact with both feet
    both_contact = (left_contact + right_contact) >= 1.5
    if both_contact:
        contact_quality = factor_vel * factor_angle   # bounded [0,1], high when stable
        w_contact = 5.0
        comp_contact = w_contact * contact_quality
    else:
        comp_contact = 0.0

    # Quadratic penalties for high velocity and attitude deviations
    w_vel_pen = 0.01
    vel_pen = -w_vel_pen * (vx**2 + vy**2)

    w_att_pen = 0.01
    att_pen = -w_att_pen * (angle**2 + angular_vel**2)

    total = comp_prox + comp_descent + comp_contact + vel_pen + att_pen

    components = {
        'proximity': comp_prox,
        'descent_quality': comp_descent,
        'contact_quality': comp_contact,
        'velocity_penalty': vel_pen,
        'attitude_penalty': att_pen,
    }
    return float(total), components
```