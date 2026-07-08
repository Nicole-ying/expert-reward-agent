# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v1 for lunar-lander-like 2D flying task.
    Components:
    1. progress_reward:   -distance_to_landing_pad (dense guidance)
    2. orientation_penalty: small penalty for non‑zero body angle and angular velocity
    3. landing_bonus:       large positive reward when lander is close, slow, and touching ground
    """
    # ---- unpack observations ----
    # current state
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    # left_contact = obs[6]   # not needed for progress
    # right_contact = obs[7]

    # next state (for landing condition)
    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    # Small coefficients so it doesn't dominate the learning signal.
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: soft proxy for successful landing ----
    # Trigger conditions:
    #   - close to the pad centre
    #   - low total speed
    #   - at least one leg in contact with ground
    close_thresh = 0.3
    speed_thresh = 0.5
    bonus_value = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5
    legs_contact = (n_left_contact > 0.5) or (n_right_contact > 0.5)

    landing_bonus = 0.0
    if n_distance < close_thresh and n_speed < speed_thresh and legs_contact:
        landing_bonus = bonus_value

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    # components dict (only includes the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）  
   - 形式：负欧氏距离 `‑distance(x_position, y_position)`。  
   - 角色：密集过程引导——每一步都给出“离目标垫越近分数越高”的连续梯度，驱动飞行器朝中央移动。  
   - 为什么用 `distance_reward` 而非 `progress_delta_reward`：`distance_reward` 提供每步状态值的绝对值，对于从静止开始的探索更友好，不会出现“不动也拿零分”的无梯度问题。

2. **orientation_penalty**（稳定约束）  
   - 形式：`‑0.05·|body_angle| ‑ 0.01·|angular_velocity|`。  
   - 角色：轻微的姿态/角速度惩罚，鼓励飞行器保持竖直，减少翻滚。权重很小，不会压制主学习信号，但能抑制无意义的剧烈旋转。

3. **landing_bonus**（任务完成近似信号）  
   - 形式：在下一步同时满足 **靠近垫中心**(距离<0.3) **且** **速度缓慢**(合成速度<0.5) **且** **至少一条支撑腿接触地面** 时，给予 +10.0 奖励。  
   - 角色：弥补“接近但不落地”的不足，给出完成软着陆的明确正向信号。由于环境没有显式 success flag，这是由多个连续条件组合的 soft proxy，符合原则 2.2 和 2.6。

## 组件预算与遵循的原则

- **role‑based component budget**：共 3 个组件，其中 1 个主信号、1 个稳定约束、1 个任务完成 proxy，符合 v1 的 2~4 个组件要求。
- **主信号命名约定**：核心学习项使用 `progress_reward` 作为 components key。
- **信号可用性优先**：只用 environment_card 声明的 obs/next_obs，不依赖 info（info 为空）。因为 explicit_success_flag_available 和 explicit_failure_flag_available 均为 false，故不使用 terminal_success_reward 和 terminal_failure_penalty。
- **稠密性与平衡**：`progress_reward` 每步稠密；`orientation_penalty` 是连续轻度惩罚；`landing_bonus` 是事件型但触发的条件需要连续变量（位置、速度）。量级上，`landing_bonus` 的 +10 能补偿初期约 -10 的距离惩罚，但只会着陆时触发一次；`orientation_penalty` 的期望值很小，不会统治学习。
- **避免信号冲突**：只使用了一个距离相关信号，没有同时使用 distance_reward 和 progress_delta。
- **阶段条件**：v1 未加入 efficiency / energy / time 惩罚，因为首要目标是学会安全着陆。
- **可利用风险防范**：`landing_bonus` 要求同时满足位置、低速、接触三个条件，防止 agent 用“高速撞击垫子”或“在远处轻触地面”来欺骗奖励。

## 未使用的组件及原因

- **terminal_success_reward / terminal_failure_penalty**：环境缺少显式成功/失败标志，info 为空，无法可靠判断终止原因。
- **距离进度增量（progress_delta）**：容易在目标附近震荡且可能造成零学习梯度，v1 改用连续的 distance_reward 作为主信号。
- **动作代价（fuel penalty）**：v1 先聚焦于完成着陆，效率类组件留到后续迭代加入，以免干扰探索。
- **复杂门控（gated_reward）**：v1 不需要阶段切换，留到后期再加入安全约束。

## 后续迭代方向与应观察的 failure mode

- **训练后应检查**：
  1. **high_reward_without_success**：agent 是否学会了在垫子正上方悬浮却不落地（`progress_reward` 高但无 `landing_bonus`）。如果出现，可考虑提高 `landing_bonus` 值或增大 `close_thresh`。
  2. **fast_crash_near_goal**：是否在接近目标时高速撞击；如果 `landing_bonus` 的 `speed_thresh` 太低可能被忽略，可调整阈值或增加速度惩罚权重。
  3. **姿态震荡**：`orientation_penalty` 是否过弱导致翻滚着陆；如有必要可略微增大系数，但要注意不要过度惩罚必要姿态修正。
- **未来迭代计划**：
  - 引入**动作代价**（小权重 engine penalty）来鼓励节省燃料。
  - 若有 wrapper 暴露成功标志，加入 terminal_success_reward 增强终点信号。
  - 进一步细化姿态/速度约束，使用 2‑norm