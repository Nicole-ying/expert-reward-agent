# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant states from the next observation (post-transition)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- Component A: Position Proximity (Main Learning Signal) ----------
    # Dense, bounded reward encouraging the craft to reach (0,0).
    # Uses soft inverse distance so that reward saturates near the target.
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    # Activation gate: high when |y| is small (near platform level), negligible in high sky.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Joint Condition Proxy) ----------
    # Soft proxy for a “successful two‑leg landing” when multiple conditions are met.
    # Each factor is a continuous [0,1] measure of a desired condition.
    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)
    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)
    k_vx = 2.0
    factor_vx = 1.0 / (1.0 + k_vx * vx_next**2)
    k_vy = 2.0
    factor_vy = 1.0 / (1.0 + k_vy * vy_next**2)
    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)
    # Contact signals are binary; the product is non‑zero only when both legs touch.
    factor_contact = left_contact * right_contact

    contact_proxy = factor_x * factor_y * factor_vx * factor_vy * factor_angle * factor_contact
    contact_reward = 1.0 * contact_proxy   # up to 1.0 when all conditions are nearly perfect

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

**任务画像**  
- task_family: `navigation_goal_reaching`  
- dynamics_subtype: `goal_approach_and_soft_contact`  
- control: discrete (4 actions, one main thruster + two lateral orientation thrusters)  

**选中的奖励角色与信号映射**  
1. **`position_proximity` (主学习信号)**  
   - 信号：`x_position`, `y_position` → 欧几里得距离的逆变换。  
   - 公式算子：`dense_state_signal` → `1/(1+dist)`，有界、每步稠密，引导飞行器飞向目标。  
2. **`soft_landing_velocity` (安全约束)**  
   - 信号：`y_position`（门控）、`x_velocity`, `y_velocity`。  
   - 公式算子：`bounded_signal` → 高度激活函数 `1/(1+10|y|)` 乘上速度平方惩罚。只在接近平台时生效，避免阻碍高空下降。  
3. **`stable_orientation` (姿态约束)**  
   - 信号：`body_angle`。  
   - 公式算子：`quadratic_penalty` → `-0.5 * angle²`。轻量惩罚，防止倾斜导致单脚着陆。  
4. **`contact_completion` (任务完成近似)**  
   - 信号：`left_support_contact`, `right_support_contact` 以及所有位置/速度/姿态信号。  
   - 公式算子：`joint_condition_proxy` → 六个连续因子的乘积，只有所有着陆条件同时趋近完美时才给出显著奖励。这些条件包括：接近原点 (x≈0, y≈0)，极低线速度，水平姿态，双腿同时接触。该奖励用 bounded 乘积实现连续梯度，避免了稀疏二值信号，可帮助策略理解“完成着陆”的含义。

**未使用的职责与原因**  
- **`fuel_efficiency`（引擎惩罚）被排除**：按 v1 阶段原则，agent 应先学会任务方向（接近并软着陆），过早加入动作代价会压制探索（agent 可能因害怕燃料惩罚而不敢使用主引擎下降）。该职责已记录，将在后续迭代中作为次要目标加入。  
- **`terminal_success_reward` / `terminal_failure_penalty`**：环境没有显式成功/失败标志（explicit_success_flag_available = false, explicit_failure_flag_available = false），且 `info` 字典为空，无法实现基于终止信号的奖励。

**未来迭代计划**  
- 在 agent 稳定着陆后，可逐步加入轻量级 `fuel_efficiency` 惩罚（例如按动作 id 给予不同的成本系数），引导最小化引擎使用。  
- 若发现接近阶段与软着陆阶段有冲突，可考虑引入 `curriculum_weighting` 调整位置奖励和速度惩罚的权重配比，但这需要训练进度信息支持，目前保留。  

**预期观察的失败模式**  
训练后应重点检查：  
- **高速撞击**：检验 episodes 终止前几步的 `y_velocity` 绝对值，若仍然很大，需加强 `soft_landing_velocity` 的权重或降低激活门的平滑度。  
- **悬停不下降**：检查 `y_position` 长期居高且动作以 no_engine 或转向为主，可能需提高 `position_proximity` 对高度接近的敏感度（例如增大距离惩罚的系数）。  
- **单脚着陆或姿态倾斜**：查看终止时是否只有一条腿接触或 `body_angle` 偏大，可能需要加大 `stable_orientation` 权重，或进一步提高 `contact_completion` 奖励对双腿接触的强调。  
- **水平漂移出视口**：若终止于 `horizontal_position_outside_viewport`，建议在后续迭代中对大 `|x|` 的区域增加位置奖励的陡峭度（例如平方距离）。  
- **旋转失控**：观察 `angular_velocity` 持续振荡，必要时可引入对其的轻量二次惩罚作为额外姿态约束，但 v1 暂不加入，避免过度约束。
