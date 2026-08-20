# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D着陆器奖励函数 v1
    使用 observation [8] 和 action (0-3) 计算奖励
    """
    # ---------- 从观测中提取变量 ----------
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 核心进展：到目标(0,0)的距离变化 ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next   # 正 = 接近目标

    # ---------- soft_health_gate：根据下一步速度与姿态衰减进展奖励 ----------
    speed_sq = vx_next**2 + vy_next**2      # 线速度平方
    speed = speed_sq**0.5
    # 门控系数：速度越高、倾角越大，gate 越小
    gate_speed = 1.0 / (1.0 + 0.5 * speed_sq)
    gate_angle = 1.0 / (1.0 + 2.0 * abs(angle_next))
    gate_health = gate_speed * gate_angle

    # progress_reward 已经带健康门控
    progress_reward = progress_delta * gate_health

    # ---------- soft_landing_contact：双腿同时接触并满足稳定条件 ----------
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        # 速度与姿态的指数衰减因子
        speed_temp = 0.2      # 速度尺度
        angle_temp = 0.1      # 角度尺度
        stable_factor = (2.718281828 ** (-speed_sq / speed_temp)) * \
                        (2.718281828 ** (-(angle_next**2) / angle_temp))
        contact_bonus = 10.0 * stable_factor   # 基础分10，由稳定性缩放

    # ---------- fuel_penalty：惩罚引擎使用 ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- orientation_stability：接近地面时额外惩罚姿态偏差 ----------
    orientation_penalty = 0.0
    if y_next < 0.5:   # 高度阈值，接近着陆垫
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    # ---------- 组合总奖励 ----------
    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    # 记录各组件，便于调试
    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**  
`navigation_goal_reaching` → `goal_approach_and_soft_contact`  
任务本质是控制着陆器从较高处降落，精确停止在目标垫上（`x≈0, y≈0`），同时要求双腿软接触、低速、竖直姿态。

**selected reward roles**  
1. **goal_proximity (主学习信号)**  
   使用 **improvement_delta**（`dist_obs - dist_next`）驱动 agent 持续向目标垫中心前进。  
   物理含义：每步距离的减少量（接近为正，远离为负），保证了每步都有梯度，且不会让 agent 停留在非目标区域。

2. **soft_landing_contact (接触 + 稳定约束)**  
   包含两个子机制：
   - **soft_health_gate**：`progress_reward * gate_health`，其中 gate 由下一步的线速度和倾角决定。速度越大、越倾斜，gate 越接近 0，从而抑制高速/歪斜状态下的进展奖励，防止“冲撞得分”。
   - **joint_condition_proxy**：当双腿同时接触垫面时给予 `contact_bonus`，用指数衰减因子乘基础分值，确保只有速度极低、角度接近竖直时才能获得高奖励。这鼓励 agent 以“软着陆”方式完成任务。

3. **fuel_penalty (燃料惩罚，次要目标)**  
   每次点燃任意引擎（`action ≠ 0`）施加小的固定负奖励，驱动燃料经济性，但不会压制必要控制动作。

4. **orientation_stability (条件姿态约束)**  
   仅在接近地面（`y_next < 0.5`）时激活，惩罚角度和角速度的平方项。避免过早限制姿态调整（下降中需要侧推机动），只在最终着陆阶段要求竖直姿态。

**role_to_signal_mapping**  
- `goal_proximity`：`obs[0,1]`, `next_obs[0,1]`（位置）→ 构造 L2 距离。
- `soft_landing_contact`（gate 部分）：`next_obs[2,3,4]`（速度、角度）→ 计算 gate。
- `soft_landing_contact`（接触 bonus）：`next_obs[6,7]`（双腿接触标志）+ `next_obs[2,3,4]`（稳定性因子）。
- `fuel_penalty`：`action` 本身。
- `orientation_stability`：`next_obs[1,4,5]`（高度、角度、角速度）→ 条件触发。

**excluded roles 及原因**  
- `success_bonus` / `terminal_state_reward`：环境无显式 success/failure flag（info 为空），禁止使用。
- `shaped_energy_limit`：缺失燃料储量信号，不可实现。
- `action_smoothness_penalty`：无上一动作历史，无法计算连续性。

**为什么没有使用 terminal_success_reward / terminal_failure_penalty**  
环境卡片明确 `explicit_success_flag_available=false` 且 `info` 为空。无法可靠推断终止原因，直接依赖会导致错误信号。当前设计通过接触稳定性 + 进展门控隐式引导安全着陆，并通过进度差惩罚阻止飞出边界。

**留到后续迭代的职责**  
- 效率/动作代价更精细建模（如主引擎与姿态引擎不同权重）。
- 更复杂的动态课程（根据高度动态调整速度惩罚权重）。
- 若未来提供剩余燃料，可加入燃料限制约束。

**训练后应观察的 failure modes**  
（与环境卡片的预期一致）
- 悬停高空：检查 `y_position` 长期较高但无下降，可能需增大 `progress_delta` 对垂直分量的权重。
- 高速撞击：着陆前速度太大导致 crash，检查 `gate_health` 中的速度系数是否需要增大。
- 反复弹跳：接触后速度未归零，`contact_bonus` 因 `stable_factor` 低而持续缺失，可考虑对连续多步接触施加额外 bonus。
- 飞出边界：`x` 绝对值急速增大，`progress_delta` 持续负值应自然抑制，若无效可单独加横向偏差惩罚。
- 姿态翻滚：`angle` 和 `angvel` 大幅波动，`orientation_penalty` 在近地时激活，若翻滚发生在高空可能需要全局轻量角速度惩罚。
