# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    obs / next_obs: 8-dim array
    [x_position, y_position, x_velocity, y_velocity, body_angle,
     angular_velocity, left_contact, right_contact]
    """
    # 当前状态
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    # 下一状态
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_av = next_obs[5]

    # 主学习信号：potential-based shaping
    # 潜力 = -distance - w_speed * speed_sq，speed_sq = vx^2 + vy^2
    # 接近且减速时潜力上升 → 正奖励
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5
    speed_sq = vx**2 + vy**2
    nspeed_sq = nvx**2 + nvy**2

    pot = -dist - 0.2 * speed_sq
    next_pot = -ndist - 0.2 * nspeed_sq

    progress_reward = 15.0 * (next_pot - pot)  # 放大系数使奖励数值明显

    # 姿态稳定约束（对下一步姿态的二次惩罚）
    # 权重较小，防止抑制必要的横向调节
    attitude_penalty = 0.5 * (n_angle ** 2) + 0.1 * (n_av ** 2)

    total_reward = progress_reward - attitude_penalty

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": -attitude_penalty,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**: navigation_goal_reaching / goal_approach_and_soft_contact
- **selected reward roles**:  
  - `goal_proximity + soft_landing_settling`（合并为一个主学习信号，通过潜力塑造同时驱动位置接近与速度控制）  
  - `orientation_stability`（轻量角/角速度惩罚，保持竖直）
- **role_to_signal_mapping**:  
  - 主信号使用 `x_position, y_position, x_velocity, y_velocity`（obs/next_obs）构造潜力函数。  
  - 姿态约束使用 `body_angle, angular_velocity`（next_obs中的值）。
- **每个 role 选择的 formula operator**:  
  - 主信号采用 `potential_based_shaping`，潜力定义为 `-distance - w_speed * speed_sq`。通过 delta 奖励同时鼓励靠近目标和减速，使接近过程中的速度自然收敛，无需单独的硬性软着陆奖励。  
  - 姿态约束采用 `quadratic_penalty`，对角度和角速度施加轻量二次惩罚。
- **excluded roles 及原因**:  
  - `soft_landing_settling` 中的接触奖励：v1 暂不依赖接触标志，避免因条件苛刻导致奖励稀疏；接近和减速已由潜力塑造覆盖。  
  - `fuel_efficiency`：离散动作下动作惩罚可能压制必要推力，v1 优先学会到达，留待后续迭代。  
  - `failure_penalty`：无法可靠检测 crash 或越界，强行使用会损害正常接近行为。  
  - `terminal_success_reward / terminal_failure_penalty`：无显式 success/failure flag，且 info 为空，完全不可用。
- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**:  
  环境中不存在显式成功/失败标志，也无法从观测中可靠推断终止原因，任何硬性的终端事件奖励都可能错误引导或破坏学习。
- **哪些职责留到后续迭代**:  
  - 基于接触标志的轻柔接触/稳定奖励（如 joint_condition_proxy）；  
  - 动作效率或燃料消耗惩罚；  
  - 更精细的软着陆门控或自适应速度约束（当基本接近策略稳定后加入）。
- **训练后应该观察哪些 failure modes**:  
  - agent 是否因潜力项中速度权重过早强调整体减速而不敢下降（表现为 y 下降缓慢）；  
  - 姿态惩罚是否导致 agent 不敢使用侧向引擎，难以修正水平偏移；  
  - 主奖励放大系数（15）是否造成每步奖励过大而导致不稳定；  
  - 在目标附近低速徘徊但不接触垫子（缺少接触奖励引导的潜在结果）。