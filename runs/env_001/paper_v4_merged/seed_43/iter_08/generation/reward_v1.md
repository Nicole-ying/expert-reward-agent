# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nl_contact, nr_contact = next_obs

    # compute distances to target (0,0)
    dist = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5

    # 1. main progress signal: improvement_delta on distance
    delta_dist = dist - dist_next  # positive when approaching target
    delta_dist_clipped = max(-0.5, min(delta_dist, 0.5))  # bound extreme jumps
    progress_reward = 1.0 * delta_dist_clipped

    # 2. speed penalty when near the target to encourage gentle approach
    close_threshold = 0.3
    if dist_next < close_threshold:
        speed_penalty = -0.5 * (abs(nvx) + abs(nvy))
    else:
        speed_penalty = 0.0

    # 3. attitude stability: hinge penalty on body angle
    safe_angle = 0.2  # radians
    angle_excess = max(0.0, abs(nangle) - safe_angle)
    angle_penalty = -0.1 * angle_excess

    # 4. fuel efficiency: small penalty for main engine usage
    fuel_cost = 0.0
    if action == 2:  # main engine
        fuel_cost = -0.02

    # 5. soft landing bonus: proxy success condition using observable signals
    success_dist_thresh = 0.1
    success_speed_thresh = 0.2
    success_angle_thresh = 0.1
    soft_landing_bonus = 0.0
    if (dist_next < success_dist_thresh and
        abs(nvx) < success_speed_thresh and
        abs(nvy) < success_speed_thresh and
        abs(nangle) < success_angle_thresh and
        nl_contact == 1 and nr_contact == 1):
        soft_landing_bonus = 10.0

    total_reward = progress_reward + speed_penalty + angle_penalty + fuel_cost + soft_landing_bonus
    components = {
        "progress_reward": progress_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "fuel_cost": fuel_cost,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family**：navigation_goal_reaching  
- **dynamics_subtype**：goal_approach_and_soft_contact  
- **selected reward roles**：
  - **progress_to_target_by_delta_distance**（主学习信号）：每步距离缩短量作为奖励，使用 `improvement_delta` 算子，避免悬停陷阱。
  - **speed_constraint_near_target**：在接近平台时用线性惩罚压降低速度，防止高速撞击，属于 `dense_state_signal(penalty)`。
  - **attitude_stability_bonus**：使用 `hinge` 算子对超出安全阈值（0.2 rad）的身体倾斜进行惩罚。
  - **fuel_efficiency_penalty**：对主引擎动作施加微小惩罚，远小于主信号，防止压制探索。
  - **soft_landing_bonus**：基于距离、速度、角度和双接触的 `joint_condition_proxy`，提供强正向信号以驱动最终着陆成功。
- **excluded roles 及原因**：
  - `soft_landing_terminal_bonus`（mandatory 中提出的不可用）—— 环境未提供 `terminated` 标志，无法在 reward 中判断终止步，故无法实现真正的终端奖励。但我们用每步可触发的 `soft_landing_bonus` 作为替代，语义上等价且可用。
  - `proximity_reward`（绝对距离奖励）—— 易导致 agent 在远处停滞仍得分，违反“尽快到达”目标，已避免。
  - `explicit_time_penalty` —— 缺乏步数信号，且可能引发早停。
  - `action_efficiency` 对离散动作空间不必要。
- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：info 为空，无显式成功/失败标志；终止状态无法在 reward 计算时获取，因此终端事件算子不可行。
- **保留到后续迭代的职责**：更精细的能耗约束、动态课程权重、基于训练进度的折扣因子等。
- **训练后应观察的主要 failure modes**：
  - **悬停而不降落**：progress_reward 的 delta 机制应使静止无奖励，但若 agent 通过来回摆动刷 delta，需关注。
  - **不敢使用主引擎**：fuel_cost 极小（–0.02），主 reward 1.0 的尺度远大于此，不应导致引擎抑制。
  - **高速撞击/翻倒**：speed_penalty 与 angle_penalty 应在前兆阶段提供梯度；若仍发生，可能需要加强速度上限或门控。
  - **着陆但 bounce**：soft_landing_bonus 要求双接触且低速低角度，可鼓励稳定着陆；若不稳可进一步强化低速奖励。