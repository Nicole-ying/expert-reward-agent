# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x = obs[0]
    y = obs[1]
    # vx, vy not directly used for reward, but could be in future iterations
    # angle, angular_vel, contacts could be used, but here we use next_obs versions

    # Unpack next observation (state after action)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ------------------  Main progress signal (improvement_delta)  ------------------
    # Reward distance reduction to the target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)  # positive when moving toward the target

    # -----------  Contact bonus: soft proxy for task completion  -----------
    # Sparse reward when at least one leg touches the pad and the lander is close to center
    w_contact = 10.0
    contact_threshold = 0.3
    contact_bonus = 0.0
    if (next_left_contact + next_right_contact) >= 1 and next_dist < contact_threshold:
        contact_bonus = w_contact

    # -------------------  Health constraint: body angle -------------------
    # Penalize extreme tilt that could lead to a crash (hinge form)
    w_angle = 0.5
    safe_angle = 0.5          # radians
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + contact_bonus + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "contact_bonus": contact_bonus,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **任务家族与动力学子类型**：`navigation_goal_reaching`，2D 飞行器着陆。目标是从上方出发，尽快且尽量少用引擎推力，降落到中央目标垫并稳定停靠。观察和动作空间均为环境卡片严格声明的结构，没有额外字段。
- **选定的奖励角色**：
  1. **主学习信号**：`progress_reward` – 通过每步距离的缩短提供稠密梯度，驱动 agent 靠近目标垫。选择 `improvement_delta` 算子（`dist - next_dist`），因为该算子始终奖励“朝目标移动”，避免 agent 在接近但不接触的状态下停滞。
  2. **任务完成近似信号**：`contact_bonus` – 当任一支撑腿接触地面且距离目标垫中心小于 0.3 时给予稀疏大额奖励。使用简单的阈值条件（`joint_condition_proxy` 的起始形式但保持单步稀疏奖励），没有伪造显式成功标志。
  3. **稳定/安全约束**：`angle_penalty` – 使用 hinge 形式惩罚超出安全阈值（0.5 rad）的身体倾斜角，避免失控翻滚。算子选自 `dense_state_signal` (hinge)，因该维度需要在越界时给出明确负向信号，而在安全范围内不惩罚，以免抑制正常机动。
- **职责-信号映射**：
  - 主进展信号 ← `obs[0], obs[1]`（水平/垂直位置），通过每一步的欧氏距离差构造。
  - 接触奖励 ← `next_obs[6], next_obs[7]`（左/右支撑腿接触标志）与 `next_dist` 结合。
  - 角度惩罚 ← `next_obs[4]`（当前身体倾斜角）。
- **排除的角色及原因**：
  - `terminal_success_reward` / `terminal_failure_penalty`：环境未提供显式 `success`/`failure` flag（`explicit_success_flag_available=false`，`explicit_failure_flag_available=false`），无法安全地依赖硬覆盖终止事件。
  - `action_efficiency`：动作空间为离散 4 选 1，暂不需连续控制代价；引入惩罚可能过早阻碍探索。
  - `soft_health_gate`：当前约束已直接惩罚危险姿态，尚未观测到需要进一步门控主奖励的失效模式。如果后续训练中出现“先冲后倒”且主奖励在失败回合仍为正的现象，可考虑将身体倾角转为 gate factor。
  - `velocity_penalty`、`dynamic_curriculum` 等：留到后续迭代加入，v1 优先让 agent 学会基本的导航降落和姿态感知。
- **为什么未使用 `terminal_success_reward` / `terminal_failure_penalty`**：观测和 `info` 字典均无确定的成功/失败标记；环境终止时可能因身体休眠（可能为成功）或碰撞/出界（失败），但这些原因不能从 8 维观测向量直接推断，因此不在 v1 中构造硬覆盖事件。
- **留待后续迭代的职责**：速度过大或下降过速的惩罚、更精细的姿态约束、动作平滑度、燃料效率优化、基于垂直速度和地面高度的 safety gate、多阶段门控奖励等。
- **训练后应观察的失败模式**：
  - **缓慢盘旋不降落**：若 progress 梯度不够强，agent 可能仅在安全高度摆动而不降低高度。需检查 `progress_reward` 的步均分量是否持续偏小。
  - **快速下冲撞击**：缺少下降速度约束可能诱发高速撞击导致崩溃；若 terminated 率高且失败回合的距离下降很快，需要后续添加 `y_velocity` 相关惩罚或 gate。
  - **接触奖励诱导的地面滑行**：如果 agent 在获得初次接触奖励后仍不断轻微移动以重复触发奖励（尽管终止逻辑应会休眠），可收紧 `contact_threshold` 或采用一次性完成奖励。
  - **倾斜侧翻**：`angle_penalty` 若过轻，agent 可能为了快速下降而忽略姿态控制；需监测 `angle_penalty` 的 episode 均值与真实 crash 率的对应关系，必要时调高 `w_angle` 或改为 gate。
