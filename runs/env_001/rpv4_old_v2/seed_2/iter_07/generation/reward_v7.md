1. `evidence`：得分从-63.97提升到-37.58，terminated从1增到3，len从963降到947，但landing_contact_success的激活率仅0.1％，说明双腿着地的成功信号几乎从未被触发。
2. `behavior_diagnosis`：策略在软着陆惩罚和距离进步的引导下缓慢接近地面，但因成功信号缺失而无法完成最终着陆，导致绝大多数回合被截断。
3. `signal_completeness`：任务需要一个可达的着陆成功代理；当前仅靠距离和速度惩罚无法区分“接近地面”与“稳定着陆”，缺少明确成功激励。
4. `selected_level`：Level 2（组件数学形式修复）。骨架方向已经使分数连续上升，着陆成功组件因稀疏导致完全死亡，需改为连续激励。
5. `selected_intervention`：唯一目标是`landing_contact_success`，将原有的稀疏速度门控bonus改为双腿接触下的连续品质奖励（距离、速度、姿态的指数乘积），确保只要双腿着地即能收到渐变信号。
6. `falsifiable_hypothesis`：施加连续着陆品质奖励后，agent将学会保持双腿接触并优化位置、速度和姿态，从而提升terminated比率、缩短episode长度并提高得分。
7. `expected_next_round`：terminated期望≥5/20，score转向正值，len降至600-800，`landing_contact_success`的active_rate显著上升，`goal_distance_progress`仍保持正向但份额下降。
8. `main_risk`：指数乘积可能在距离、速度或角度较大时给出极低奖励，若初始探索没得到任何接触奖励，进展可能缓慢；但只要有短暂接触就能获得非零信号，风险可控。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous proximity-gated quality)
    w_success = 5.0
    k_dist_success = 5.0
    k_speed = 2.0
    k_angle_success = 5.0
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    contact_flag = 1.0 if legs_on_ground else 0.0
    dist = (next_x**2 + next_y**2) ** 0.5
    speed = (next_vx**2 + next_vy**2) ** 0.5
    angle_abs = abs(next_angle)
    quality = (2.718281828 ** (-k_dist_success * dist)) * \
              (2.718281828 ** (-k_speed * speed)) * \
              (2.718281828 ** (-k_angle_success * angle_abs))
    contact_bonus = w_success * contact_flag * quality

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```