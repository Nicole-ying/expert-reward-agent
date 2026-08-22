# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    LunarLander-style discrete control environment.
    Obs: [x, y, vx, vy, angle, angvel, left_contact, right_contact]
    Goal: reach target pad (x≈0,y≈0) with both legs down, low speed, upright attitude.
    """
    # Unpack observations
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    angvel = obs[5]
    left = obs[6]
    right = obs[7]

    # ---------- helper quantities ----------
    distance = (x**2 + y**2) ** 0.5
    speed = (vx**2 + vy**2) ** 0.5

    # ---------- 1. Goal proximity (main driving signal) ----------
    w_prox = 1.0
    proximity_penalty = -w_prox * distance

    # ---------- 2. Velocity control (gated, near target) ----------
    w_vel = 0.5
    d_threshold = 0.5
    gate = max(0.0, 1.0 - distance / d_threshold)   # active when close
    vel_penalty = -w_vel * (vx**2 + vy**2) * gate

    # ---------- 3. Attitude stabilization ----------
    w_angle = 0.5
    w_angvel = 0.1
    attitude_penalty = -w_angle * (angle**2) - w_angvel * (angvel**2)

    # ---------- 4. Stable contact proxy ----------
    # Smooth reward for both legs touching, maximised when close & slow
    w_contact = 5.0
    kd = 5.0            # distance sensitivity
    kv = 10.0           # speed sensitivity
    proximity_factor = 1.0 / (1.0 + kd * distance)
    speed_factor = 1.0 / (1.0 + kv * speed)
    contact_bonus = w_contact * left * right * proximity_factor * speed_factor

    # ---------- Assemble total reward ----------
    total_reward = proximity_penalty + vel_penalty + attitude_penalty + contact_bonus

    components = {
        "proximity_penalty": proximity_penalty,
        "vel_penalty": vel_penalty,
        "attitude_penalty": attitude_penalty,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**  
  `navigation_goal_reaching` / `goal_approach_and_soft_contact`。主目标是到达目标垫并稳定双足接触，隐含安全与节能要求。

- **selected reward roles**  
  - `goal_proximity` – 主学习信号  
  - `velocity_control` – 安全/稳定约束  
  - `attitude_stabilization` – 姿态稳定约束  
  - `stable_contact` – 任务完成软近似（没有显式 success flag）

- **role_to_signal_mapping**  
  - `goal_proximity`  → `obs[0]`, `obs[1]` (位置)  
  - `velocity_control` → `obs[2]`, `obs[3]` (速度)，用 `obs[0]`, `obs[1]` 的距离做门控  
  - `attitude_stabilization` → `obs[4]`, `obs[5]`  
  - `stable_contact` → `obs[6]`, `obs[7]` (接触标志)，结合距离和速度做连续门

- **每个 role 选择的 formula operator**  
  - `goal_proximity`：`distance_penalty` (负欧几里得距离)，提供每步梯度  
  - `velocity_control`：带线性门控的 `quadratic_penalty`，只在接近目标时激活  
  - `attitude_stabilization`：`quadratic_penalty` 分别惩罚机体角与角速度  
  - `stable_contact`：乘积形式的连续 `joint_condition_proxy`，用 `1/(1+k*error)` 对距离和速度做光滑因子，避免稀疏

- **excluded roles 及原因**  
  - `fuel_efficiency`：v1 阶段让 agent 先学会“飞向目标并安全着陆”，暂不限制引擎使用，避免抑制必要的机动探索  
  - `fast_settling_incentive`：需要训练中后期引入，且缺少显式时间信号，v1 不宜加入  
  - `terminal_success_reward` / `terminal_failure_penalty`：`explicit_success_flag_available=false` 且 `explicit_failure_flag_available=false`，info 不可用，因此无法设计可靠的终止事件奖励  
  - 没有使用 `potential_based_shaping`，因为距离本身已经提供了直接的进展度量

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
  环境卡片明确说明不存在显式成功/失败标志，info 为空，且终止条件仅能从观测中模糊推断。强行构造硬终止奖励会引入较大噪声和偏差，因此采用连续近似（如 `stable_contact` 作为软 success proxy）。

- **哪些职责留到后续迭代**  
  - 引擎燃料效率（`fuel_efficiency`）—— 在 agent 能稳定到达后加入  
  - 快速沉降激励（`fast_settling_incentive`）—— 当任务完成率提升后通过时间惩罚加速收敛  
  - 可能加入更复杂的健康门控或动态权重，以应对速度/姿态的早发失效

- **训练后应该观察哪些 failure modes**  
  - 无法接近目标（远处徘徊）：检查距离惩罚是否足够，考虑提升 `w_prox` 或使用 potential shaping  
  - 高速撞击或反弹：加强 `velocity_control` 权重或降低门控阈值  
  - 单腿着地后翻倒：增加 `contact_bonus` 权重，同时观察姿态惩罚是否抑制翻倒  
  - 在错误位置过早“settled”终止：因为环境可能因静止而终止，若远离目标需要强化位置条件（当前 design 依赖 proximity 间接避免，后续可考虑更显式的位置门控）  
  - 过量引擎使用但无法到达：后续加入燃料惩罚，同时确保距离信号能引导高效路径