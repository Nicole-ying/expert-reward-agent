# Response Record

1. `evidence`：current_score=-26.30（best但远低于target 200），ep_len=951接近上限，terminated仅4/20、truncated=16/20，landing_reward占据magnitude_share 93.2%且active_rate 82.7%，mean episode_sum≈1523——策略在地面附近长时间维持良好姿态但未真正完成着陆，上一轮已从-70提升至-26但结构缺陷未修复。
2. `behavior_diagnosis`：策略学会在y<0.5区域保持双脚触地、低速、小角度来持续收割landing_reward（每步约+1.6），但始终未进入settled终止状态，属于典型的持久状态奖励proxy farming，而非完成着陆任务。
3. `signal_completeness`：proximity提供弱目标引导（100% active，~0.07/步），fuel_penalty抑制引擎滥用，landing_reward意图引导着陆但以绝对状态值发放——数学形态允许无限累积而无需完成任务，缺少“改进而非占据”的落地信号。
4. `selected_level`：Level 2 `state_to_improvement` —— landing_reward职责方向正确但数学形态为持久状态值，证据显示被大规模farming（1523/episode但truncated率80%），需将绝对值变换为势能差。
5. `selected_intervention`：唯一修改`landing_reward`，从绝对状态质量`contact_score+speed_score+angle_score`变为势能差`2.0 * (potential(next_obs) - potential(obs))`，势能函数与原质量评分相同。进入着陆区时获得正向改善奖励，维持稳态得零，退化受罚，系数2.0匹配新值域。
6. `falsifiable_hypothesis`：势能差形式取消了对“占据好状态”的持续奖励，策略无法通过悬停收割reward，唯一正向来源是改善着陆质量或通过proximity靠近目标，应促使策略快速着陆后进入settled终止，减少truncated比例。
7. `expected_next_round`：episode_length应显著缩短（不再悬停），terminated比例上升（正常着陆后settle），landing_reward的episode_sum_mean大幅下降至5-20量级，score应向正值移动，active_rate反映过渡阶段而非持续悬停。
8. `main_risk`：势能差在早期高空阶段（y>>0.5）恒为零，仅靠proximity引导可能下降过慢或探索不足；若着陆后episode未及时终止，proximity的+0.1/步可能成为新的微弱farming源，但远弱于当前问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state
    prev_y = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_angle = obs[4]
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]

    # Next state
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # 1. Proximity: unchanged state-based bounded reward for being near target center
    dist = (next_x ** 2 + next_y ** 2) ** 0.5
    proximity = 0.1 / (1.0 + dist)

    # 2. Landing reward: state_to_improvement transform
    #    old: persistent absolute landing quality -> farmed by hovering
    #    new: potential difference rewards improvement, zero for maintaining
    def landing_potential(y, left_c, right_c, xv, yv, angle, threshold=0.5):
        if y < threshold:
            contact_score = left_c + right_c
            total_speed = abs(xv) + abs(yv)
            speed_score = max(0.0, 1.0 - total_speed / 2.0)
            angle_score = max(0.0, 1.0 - abs(angle) / 0.5)
            return contact_score + speed_score + angle_score
        return 0.0

    prev_pot = landing_potential(prev_y, prev_left_contact, prev_right_contact,
                                 prev_x_vel, prev_y_vel, prev_angle)
    curr_pot = landing_potential(next_y, next_left_contact, next_right_contact,
                                 next_x_vel, next_y_vel, next_angle)

    landing_reward = 2.0 * (curr_pot - prev_pot)

    # 3. Fuel efficiency penalty: unchanged
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = proximity + landing_reward + fuel_penalty

    components = {
        "proximity": proximity,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```
