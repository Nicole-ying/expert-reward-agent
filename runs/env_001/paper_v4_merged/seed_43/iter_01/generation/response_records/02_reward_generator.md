# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current observation
    x = float(obs[0])
    y = float(obs[1])
    vx = float(obs[2])
    vy = float(obs[3])
    # obs[4] is current body_angle – not used separately to avoid double penalizing,
    # penalty is applied only on next state

    # Next observation
    nx = float(next_obs[0])
    ny = float(next_obs[1])
    nvx = float(next_obs[2])
    nvy = float(next_obs[3])
    nangle = float(next_obs[4])

    # ---------- 1. Main learning signal: potential-based shaping ----------
    # Potential: Phi(s) = -(distance_from_target + alpha * speed)
    # Shaping reward = Phi(s') - Phi(s)
    # This encourages reducing distance AND slowing down when near the target,
    # giving dense gradients every step.
    dist_obs = (x * x + y * y) ** 0.5
    dist_next = (nx * nx + ny * ny) ** 0.5
    speed_obs = (vx * vx + vy * vy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5

    alpha = 0.5          # trade-off between position and speed
    potential_obs = -(dist_obs + alpha * speed_obs)
    potential_next = -(dist_next + alpha * speed_next)
    progress_shaping = potential_next - potential_obs

    # ---------- 2. Stability constraint: body angle hinge ----------
    # Penalise only when the tilt exceeds a safe threshold (in radians).
    # Small tilts are unrestricted, promoting exploration without continuous penalty.
    angle_threshold = 0.3       # ~17 degrees
    angle_hinge = -0.5 * max(0.0, abs(nangle) - angle_threshold)

    # ---------- 3. Efficiency bonus: action penalty ----------
    # Discourage unnecessary engine use: any non‑zero action incurs a tiny cost.
    action_cost = -0.01 * (0.0 if action == 0 else 1.0)

    total_reward = progress_shaping + angle_hinge + action_cost

    components = {
        "progress_shaping": progress_shaping,
        "angle_hinge": angle_hinge,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype:**  
  `navigation_goal_reaching` with `goal_approach_and_soft_contact`.  
  目标是将 2D 飞行器精确降落到目标垫中心并稳定停靠，次要要求为节能。

- **selected reward roles:**  
  1. **主学习信号** – 同时驱动接近目标和减速。  
  2. **稳定/安全约束** – 限制机身倾角，防止翻倒。  
  3. **效率约束** – 轻微惩罚非必要引擎使用。

- **role_to_signal_mapping:**  
  - **进度/减速**（`progress_shaping`）：  
    `x_position`, `y_position` → 距离；  
    `x_velocity`, `y_velocity` → 速率。  
  - **倾角约束**（`angle_hinge`）：  
    `next_obs[4]`（机身倾角）。  
  - **动作代价**（`action_cost`）：  
    `action` 是否为 0。

- **每个 role 选择的 formula operator:**  
  - 主信号使用 **potential_based_shaping**，将距离与速率组合成一个标量 potential，利用差分给出每步稠密梯度。同时解决“高速撞垫”的漏洞。  
  - 倾角使用 **dense_state_signal (hinge)**，仅在倾角超出安全阈值时才施加惩罚，避免全时压制小幅姿态调整。  
  - 动作代价采用简化的离散 **action_efficiency** 惩罚。

- **excluded roles 及原因:**  
  - `terminal_success_reward` / `terminal_failure_penalty` – 环境没有显式成功/失败标志，`info` 为空，且视口边界与地面高度未知，无法可靠推断终止事件。  
  - `soft_health_gate` / `joint_condition_proxy` – 在 v1 保持简单，优先让策略学会靠近并减速，后续再引入“双脚着垫”软奖励。  
  - `angular_velocity` 约束 – 暂不加入，倾角惩罚已足够抑制大幅度旋转。

- **为什么没有使用 terminal_success_reward / terminal_failure_penalty:**  
  `explicit_success_flag_available=false`, `explicit_failure_flag_available=false`，且 `info` 字典没有任何终止原因字段。从观测推断成功/失败需要大量环境边界知识、阈值猜测和拟合，误差大且脆弱，因此 v1 完全避免这类硬奖励。

- **哪些职责留到后续迭代:**  
  - “双脚同时着垫”的软完成代理（需要连续 bounded factor 组合而不使奖励坍缩）。  
  - 接近地面时的垂直速度安全门控（防止高速砸向目标垫）。  
  - 更精细的 angular velocity 约束或姿态稳定性 gate。  
  - 训练进程驱动的动态权重（curriculum）或动作平滑性。

- **训练后应该观察哪些 failure modes:**  
  - **犹豫/徘徊**：agent 靠近目标但不敢 finalize 触地，长时间悬停。  
  - **保守冲量不足**：action_cost 虽小但可能抑制引擎使用，导致下降过慢。  
  - **利用倾角阈值**：习惯将倾角保持在接近 0.3 rad 的边界以获得高 progress 奖励而忽略稳定着陆。  
  - **速度控制失效**：alpha 选择不当可能导致快速冲撞仍获取高总分（若 alpha 太小）。  
  - **未精确着垫**：未设计脚接触奖励，可能只停在目标上空而未触发成功判定（但物理环境会自动稳定），需观察是否需要对脚接触额外引导。
