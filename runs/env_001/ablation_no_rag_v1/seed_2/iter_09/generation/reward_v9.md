1. `evidence`: 外部score为-150.57，所有episode均为时间截断(1000步)，无任何终止条件触发；组件landing_reward极高(1058.59, 99.3% share)说明agent在高密度接近奖励下徘徊但未触发双脚触地等最终着陆条件；历史最佳迭代7使用稀疏乘积结构得分0.62并触发早期终止，本轮dense替换导致proxy exploitation。
2. `behavior_diagnosis`: 策略利用dense landing guidance长时间停留在接近中心的悬停状态获取大量奖励，未完成双脚接触/零速等最终着陆要求，无法触发成功终止，原始任务得分极低。
3. `signal_completeness`: 当前奖励缺失对齐任务完成(双脚接触+零速+位置姿态达标)的信号，稠密接近引导被过度利用，必要完成职责不可达。
4. `selected_level`: Level 2 —— 证据模式为“proxy提高但外部任务不升”，触发proxy_to_completion_alignment变换。
5. `selected_intervention`: 将landing_reward从稠密加权和的proximity形态恢复为稀疏乘积形态(与best相同)，仅在有双脚接触时才能获得正奖励，并将系数k_landing由5.0提升至20.0以放大正确行为信号。
6. `falsifiable_hypothesis`: 恢复接触门控后，agent无法再通过悬停获得高回报，必须主动完成双脚着陆才能获取奖励，应重新触发成功终止并提升外部得分。
7. `expected_next_round`: 下一轮episode_length显著缩短(出现成功终止)，external score由负转正并超越历史最佳(0.62)，landing_quality_reward的active_rate下降但非零，potential_diff引导仍保持正向。
8. `main_risk`: 大幅提高完成奖励可能造成agent急于着地而忽略姿态稳定或速度控制，angle_penalty相对不足，可能引入硬着陆或姿态振荡。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress (unchanged) --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty (unchanged) --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Sparse landing completion reward (restored to product with contact gate) --
    # Gives reward only when both legs are in contact AND all final-state conditions are met.
    # Each dimension uses max(0, 1 - abs(error)/threshold) for a smooth [0,1] signal.
    x_ok = max(0.0, 1.0 - abs(x_n) / 0.2)
    y_ok = max(0.0, 1.0 - abs(y_n) / 0.3)
    vx_ok = max(0.0, 1.0 - abs(vx_n) / 0.2)
    vy_ok = max(0.0, 1.0 - abs(vy_n) / 0.2)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_both = left_contact_n * right_contact_n   # 0.0 or 1.0

    landing_quality = contact_both * x_ok * y_ok * vx_ok * vy_ok * angle_ok
    k_landing = 20.0
    landing_quality_reward = k_landing * landing_quality

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + landing_quality_reward

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "landing_quality_reward": float(landing_quality_reward)
    }

    return float(total_reward), components
```