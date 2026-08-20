# Response Record

1. evidence：iter_07（w_vel=1.0）得分156.7、终止16/20、着陆接触激活52.5%；iter_08仅将w_vel提至2.0即见分数暴跌至-18.99、长度1000、终止0/20、着陆接触激活率0%。  
2. behavior_diagnosis：iter_07的策略学会了接近并着陆；iter_08速度惩罚过重，代理为避免惩罚在远端徘徊，始终未能触发着陆奖励，导致总分断崖式下跌。  
3. signal_completeness：当前信号包含距离进步、近垫速度抑制、姿态惩罚和着陆接触奖励，职责完备；但远场速度抑制的gating强度可能仍是接近的阻力，需更精准地仅在近垫生效。  
4. selected_level：Level 2（修改单一组件公式），soft_landing的proximity_gate从反二次型改为指数衰减，以缩小远场惩罚保持w_vel=1.0。  
5. selected_intervention：唯一修改soft_landing组件，将`proximity_gate = 1.0/(1.0 + k_dist * dist_next**2)`换为`proximity_gate = 2.718281828 ** (-k_dist * dist_next)`，k_dist=10.0。  
6. falsifiable_hypothesis：指数衰减门控只在高距程时几乎清零速度惩罚，允许代理快速接近垫，并在接近垫时强制减速，应能恢复并可能超越iter_07的成功着陆表现。  
7. expected_next_round：score回升至正值、terminated episodes数>0、landing_contact_success active_rate>0、goal_distance_progress保持正向、平均长度远小于1000。  
8. main_risk：指数门控衰减过快可能令代理在垫附近仍保持较大速度，来不及减速导致坠毁，score可能再次下降。

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

    # 2. Soft landing (exponential proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 10.0
    proximity_gate = 2.718281828 ** (-k_dist * dist_next)
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
