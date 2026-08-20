# Response Record

1. `evidence`：iter 9 最终评估显示 episode 长度 1000（截断）、terminated 0/20，landing_contact_success=0，累计 goal_distance_progress=6.28 但 orientation_penalty=-1.59、soft_landing=-0.64 导致总分为-12.57；而最佳 iter 7 得分为 156.7、len=578，说明当前版本的 soft_landing/landing 修改破坏了成功的着陆行为。
2. `behavior_diagnosis`：策略学会了在没有着陆奖励时悬停或漂移，回避着陆，持续至最大步数截断，并累积大量的速度/姿态惩罚，导致总分大幅偏离目标。
3. `signal_completeness`：已有进度、接近减速、姿态约束和着陆品质信号，但缺少对发动机使用的惩罚，可能放任无谓悬停延长 episode；补上燃料效率信号有助于缩短成功着陆时间。
4. `selected_level`：Level 2（在最佳骨架上添加组件），因为 iter 7 的骨架已经能够成功着陆，只需引入推力惩罚激励更快终止。
5. `selected_intervention`：回到 iter 7 最佳代码，新增 `thrust_penalty = -0.05 * float(action != 0)`，惩罚任何非零动作，鼓励必要之外的节能。
6. `falsifiable_hypothesis`：添加小幅推力惩罚后，代理将减少悬停，加速稳定着陆，从而缩短 episode 并减少累积负分，得分应提升。
7. `expected_next_round`：score 应上升，len 应下降（更多 successful termination），landing_contact_success 应重新非零并发挥正面作用。
8. `main_risk`：推力惩罚可能妨碍减速或姿态调整所需的发动机点火，导致接近阶段速度控制变差甚至碰撞，系数需适中。

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

    # 5. Thrust penalty (fuel efficiency)
    w_thrust = 0.05
    thrust_penalty = -w_thrust * float(action != 0)

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus + thrust_penalty

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus,
        "thrust_penalty": thrust_penalty
    }

    return float(total_reward), components
```
