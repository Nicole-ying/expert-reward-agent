# Response Record

1. `evidence`：上一轮添加的200稀疏成功奖励活跃率仅0.7%，却贡献了99.3%的比例，导致常规信号塌缩，score从-24.46暴跌至-112.79，表明稀疏事件未被有效学习。
2. `behavior_diagnosis`：agent在势能差引导下能接近目标区域，但缺少接触与精细姿态的局部反馈，无法稳定触发成功终止，策略处于负分徘徊。
3. `signal_completeness`：缺少从“接近”到“完成”的连续完成信号，只靠势能差和角度惩罚无法区分接触成功与非成功静止，有必要增加稠密完成指示。
4. `selected_level`：Level 2，触发条件为sparse_to_dense——上一轮稀疏bonus失败，且任务需要基于接触、位置、速度、角度的连续完成证据。
5. `selected_intervention`：移除success_bonus，新增landing_quality_reward（双脚接触时对位置、速度、角度按乘积阈值输出连续值，系数5.0）。
6. `falsifiable_hypothesis`：连续landing_quality能在接近目标后每步提供正向反馈，使agent学习精细着陆动作，避免稀疏bonus的信用分配困难，从而提升得分。
7. `expected_next_round`：landing_quality_reward的episode_sum_mean应为正值且active率>10%，score应回升并超过-20，episode_length可能缩短。
8. `main_risk`：乘积条件可能仍过于严格，导致初期的active率仍低，但可通过放宽阈值或提高系数解决。

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

    # -- Component 1: Potential-based progress --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Dense landing quality reward --
    # Provides per-step reward when both legs touch the ground and the pose is near-ideal.
    # All conditions use a linear decay toward zero outside the thresholds.
    x_ok = max(0.0, 1.0 - abs(x_n) / 0.2)
    y_ok = max(0.0, 1.0 - abs(y_n) / 0.3)
    vx_ok = max(0.0, 1.0 - abs(vx_n) / 0.2)
    vy_ok = max(0.0, 1.0 - abs(vy_n) / 0.2)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_both = left_contact_n * right_contact_n   # 0.0 or 1.0

    landing_quality = contact_both * x_ok * y_ok * vx_ok * vy_ok * angle_ok
    k_landing = 5.0
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
