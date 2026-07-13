# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation to a target platform.
    Mandatory roles:
      - progress_to_goal: negative Euclidean distance to target
      - successful_settle: joint condition proxy (close, slow, upright, both legs on platform)
      - engine_efficiency: penalty for any engine thrust
    Conditional role:
      - orientation_penalty: gated by distance, penalises angle and angular velocity
    """
    # Current state (not used much, but available)
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    # Next state (result of action)
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Component A: progress_to_goal ---
    # Negative Euclidean distance (high when far, near zero when close)
    distance = (nx ** 2 + ny ** 2) ** 0.5
    w_dist = 1.0
    progress_reward = -w_dist * distance

    # --- Component B: successful_settle (joint condition proxy) ---
    # Each factor ranges [0,1]; product encourages simultaneous satisfaction.
    # proximity: high when close
    k_prox = 5.0
    proximity_factor = 1.0 / (1.0 + k_prox * distance)

    # velocity: high when speed is low
    k_vel = 5.0
    speed_sq = nvx ** 2 + nvy ** 2
    velocity_factor = 1.0 / (1.0 + k_vel * speed_sq)

    # angle: high when upright
    k_angle = 4.0
    angle_factor = 1.0 / (1.0 + k_angle * abs(nangle))

    # contact: encourages both legs in contact
    contact_factor = 0.5 * (nleft_contact + nright_contact)

    w_settle = 10.0
    settle_proxy = w_settle * proximity_factor * velocity_factor * angle_factor * contact_factor

    # --- Component C: orientation_penalty (gated by distance) ---
    # gating weight: 1 when distance=0, decays as distance grows
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * distance)

    w_orient = 1.0
    w_angvel = 0.2
    orientation_penalty = -w_orient * gate * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # --- Component D: engine_efficiency ---
    # Penalise any thrust (action != 0)
    w_engine = 0.1
    engine_penalty = 0.0
    if action != 0:
        engine_penalty = -w_engine

    # Total reward
    total_reward = progress_reward + settle_proxy + orientation_penalty + engine_penalty

    components = {
        "progress_to_goal": float(progress_reward),
        "successful_settle_proxy": float(settle_proxy),
        "orientation_penalty": float(orientation_penalty),
        "engine_efficiency": float(engine_penalty),
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype:**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`  
  要求飞行器飞向目标并安全、稳定着陆（双腿接触、姿态直立、速度接近零）。

- **selected reward roles:**  
  - `progress_to_goal`（主学习信号）  
  - `successful_settle`（必要，通过联合条件代理实现）  
  - `engine_efficiency`（必要，惩罚引擎使用）  
  - `orientation_penalty`（条件角色，仅在靠近目标时加强）  

  这4个角色覆盖了环境卡片规定的全部`mandatory_roles`以及推荐的条件角色，且总数符合v1的组件预算（2~4个）。

- **role_to_signal_mapping:**  
  - `progress_to_goal` ← `nx, ny`（负欧氏距离，`dense_state_signal`）  
  - `successful_settle` ← `nx, ny, nvx, nvy, nangle, nleft_contact, nright_contact`（`joint_condition_proxy`）  
  - `engine_efficiency` ← `action`（`action_penalty`）  
  - `orientation_penalty` ← `nangle, nang_vel, nx, ny`（`quadratic_penalty` + 距离门控）

- **每个role选择的formula operator:**  
  - `progress_to_goal`: `dense_state_signal`（负距离惩罚）  
  - `successful_settle`: `joint_condition_proxy`（位置、速度、角度、接触四个连续因子的乘积）  
  - `engine_efficiency`: `action_penalty`（离散动作惩罚）  
  - `orientation_penalty`: `quadratic_penalty` + `gated_by_progress`（距离门控的二次惩罚）

- **excluded roles 及原因:**  
  - `terminal_success_reward` / `terminal_failure_penalty`：环境没有显式成功/失败标志（`explicit_success_flag_available=false`），不能使用。  
  - `time_to_goal_penalty`：环境不提供步数或时间信号，无法设计。  
  - `survival_bonus`：与快速着陆目标冲突，且无可用生存度量。  
  - 全局`velocity_smoothing`：会阻碍早期快速接近目标，仅在条件角色中通过距离门控引入。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty:**  
  环境卡片明确指出`explicit_success_flag_available=false`且`explicit_failure_flag_available=false`。所有成功/失败判断只能从观测状态推断。因此，v1采用稠密的`joint_condition_proxy`引导着陆，避免了稀疏、不可靠的终端奖励。

- **哪些职责留到后续迭代:**  
  - 双腿对称接触的明确奖励（鼓励双落腿）  
  - 更精细的燃料效率曲线（连续动作代价或动作平滑性）  
  - 动态课程权重（若训练进度显示早期探索与后期精细控制冲突）  
  - 对反弹或翻滚的额外惩罚（需要更复杂的着陆稳定性检测）

- **训练后应该观察的 failure modes:**  
  - 飞行器悬停不动，仅依赖动作惩罚导致策略保守 → 检查`progress_to_goal`权重是否过弱。  
  - 高速撞击平台（crash）→ 检查`velocity_factor`是否足够抑制下降速度，或增加`orientation_penalty`。  
  - 着陆后弹跳或翻滚（接触闪烁）→ 可能需要强化`successful_settle_proxy`中的速度阈值。  
  - 单腿着陆倾斜不稳 → `contact_factor`仅为平均接触，可能还需加大`angle_factor`或引入不对称惩罚。  
  - 长期使用主引擎但无进步 → 检查距离奖励是否足够，或降低`engine_penalty`权重。
