# Search objective
- target_score: 200.000000
- current_score: -110.442116
- gap_to_target: 310.442116
- target_achievement_ratio: -55.221%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -110.442116）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Gym-style reward designed for 2D vehicle trajectory optimization.
    Focus: proximity + gated stability + continuous velocity control + contact proxy.
    """
    # --- Unpack states ---
    x_pos, y_pos, x_vel, y_vel, body_angle, ang_vel, left_contact, right_contact = obs
    nx_pos, ny_pos, nx_vel, ny_vel, nbody_angle, nang_vel, nleft_contact, nright_contact = next_obs

    # Constants
    current_sq_dist = x_pos ** 2 + y_pos ** 2
    next_sq_dist = nx_pos ** 2 + ny_pos ** 2

    PROXIMITY_SCALE = 10.0
    ANGLE_GATE_K = 15.0
    VEL_PENALTY_COEFF = 0.02
    GROUND_PROXIMITY_SCALE = 4.0
    CONTACT_PROXY_K = 4.0
    FUEL_COST_WEIGHT = 0.15

    # 1. Goal Proximity (mandatory)
    dist_improvement = current_sq_dist - next_sq_dist
    proximity_reward = PROXIMITY_SCALE * dist_improvement / (1.0 + abs(dist_improvement))

    # 2. Orientation Stability Gate
    angle_error = abs(nbody_angle)
    angle_gate = 2.718281828 ** (-ANGLE_GATE_K * angle_error ** 2)
    gated_proximity = proximity_reward * angle_gate

    # 3. Continuous Velocity Control (penalizes speed, stronger near ground)
    speed = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    ground_proximity = 1.0 / (1.0 + GROUND_PROXIMITY_SCALE * abs(ny_pos))
    velocity_penalty = -VEL_PENALTY_COEFF * (speed ** 2) * ground_proximity

    # 4. Soft Landing Contact Proxy (rewards contact near ground, no speed condition)
    leg_contact_both = min(nleft_contact, nright_contact)
    near_ground_factor = 1.0 / (1.0 + CONTACT_PROXY_K * abs(ny_pos))
    contact_proxy = (leg_contact_both * near_ground_factor) ** 0.5
    soft_landing_bonus = 2.0 * contact_proxy

    # 5. Action Efficiency
    is_engine_on = 0.0 if action == 0 else 1.0
    fuel_cost = -FUEL_COST_WEIGHT * is_engine_on

    total_reward = gated_proximity + velocity_penalty + soft_landing_bonus + fuel_cost

    components = {
        "proximity_reward": gated_proximity,
        "velocity_penalty": velocity_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "fuel_cost": fuel_cost
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=-110.442116, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746612, -95.059093]

## Reward component values (mean per episode)
- proximity_reward: 17.906316
- soft_landing_bonus: 2.137287
- velocity_penalty: -0.700594
- fuel_cost: -0.382500

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个2D飞行器轨迹优化任务。飞行器从视口顶部中央附近出发，受到一个初始随机力的干扰。核心目标是尽快到达中央的目标着陆板并稳定停靠（低速度、正姿态、双足接触）。次要目标是尽量节省引擎推力。智能体必须学会在有限的离散推力动作下，平顺地接近目标、减速、保持姿态稳定，并实现安全的 soft landing 接触。不应把生存、保持平衡或无关区域的探索当作主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: `x_position`，相对于目标板的水平坐标，reward_usable: true
- obs[1]: `y_position`，相对于目标板高度的垂直坐标，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体倾斜角度，reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左足接触标志（0.0/1.0），reward_usable: true
- obs[7]: `right_support_contact`，右足接触标志（0.0/1.0），reward_usable: true

所有 obs 维度均可用于奖励计算，但必须在适当上下文中使用（例如 contact 只在接近地面时才有意义）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine` — 不启动任何引擎，仅依靠惯性/重力
- action 1: `left_orientation_engine` — 启动一个侧向姿态引擎（产生旋转推力）
- action 2: `main_engine` — 启动主引擎（产生主要向上的推力，可能同时影响旋转）
- action 3: `right_orientation_engine` — 启动相反侧的姿态引擎

动作本身是离散的，不能直接输出连续推力大小。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` 中的 `body_settled` 部分（如果环境以此表征成功着陆并稳定）. 任务的期望结束状态是低速、双足接触、位置在目标板附近且机体稳定，所以达到这种状态并触发“settled”应视为成功。
- failure-like termination:
    - `crash_or_body_contact` — 机体与障碍物碰撞或身体非正常触地（除目标板外的接触）
    - `horizontal_position_outside_viewport` — 水平位置超出允许边界
- ambiguous termination: `body_not_awake_or_settled` 中的 `body_not_awake` 可能表示机体翻转或失去平衡（无法再唤醒），也可能是安全停止；但描述中“not awake or settled”暗示两种情形，其中 `body_not_awake` 更可能是不良终止（如翻倒失去动力），而 `body_settled` 是成功。需根据实际环境验证。
- truncation: 无明确时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （空字典，没有字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何隐藏的 success、failure、termination_reason

**重要**：由于 info 为空且禁止推断其内容，奖励函数不能直接依赖终止原因；只能通过 `obs` 和 `next_obs` 中的状态来启发式地判断接近成功（如接近目标、低速、接触）或惩罚失败（如坠毁前的剧烈状态变化），但不能假定一定能访问终止原因。

## 7. 可用于奖励函数的信号
- position: `x_position`, `y_position`
- velocity: `x_velocity`, `y_velocity`
- orientation: `body_angle`
- angular_velocity: `angular_velocity`
- contact: `left_support_contact`, `right_support_contact`
- action/engine: 通过动作选择可推论引擎使用情况（动作 0 为 no_engine，其他为使用引擎）
- other: 无其他

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_cost + progress + soft_landing | -285.17 | -285.17 | 0.00 | 86.15 | fuel_cost=-0.008 progress=0.008 soft_landing=-0.004 | new_best |
| 2 | fuel_cost + progress + soft_landing | -298.17 | -285.17 | -13.00 | 79.80 | fuel_cost=-0.008 progress=0.009 soft_landing=-0.001 | no_meaningful_improvement |
| 3 | fuel_cost + progress + soft_landing | -125.27 | -125.27 | 0.00 | 78.20 | fuel_cost=-0.005 progress=0.016 soft_landing=0.019 | new_best |
| 4 | angle_penalty + contact_bonus + fuel_cost + progress + vertical_penalty | -122.56 | -122.56 | 0.00 | 68.30 | angle_penalty=-0.000 contact_bonus=0.002 fuel_cost=-0.001 progress=0.016 vertical_penalty=-0.018 | new_best |
| 5 | angle_penalty + contact_bonus + fuel_cost + progress + vertical_penalty | -124.81 | -122.56 | -2.24 | 68.30 | angle_penalty=-0.000 contact_bonus=0.002 fuel_cost=-0.001 progress=0.016 vertical_penalty=-0.013 | no_meaningful_improvement |
| 6 | angle_penalty + contact_bonus + fuel_cost + progress + vertical_penalty | -122.02 | -122.02 | 0.00 | 68.30 | angle_penalty=-0.000 contact_bonus=0.002 fuel_cost=-0.001 progress=0.016 vertical_penalty=-0.009 | unsolved_stagnation_fresh_restart |
| 7 | fuel_cost + proximity_reward + soft_landing_bonus + velocity_penalty | -110.82 | -110.82 | 0.00 | 68.45 | fuel_cost=-0.013 proximity_reward=0.253 soft_landing_bonus=0.022 velocity_penalty=0.000 | new_best |
| 8 | fuel_cost + proximity_reward + soft_landing_bonus + velocity_penalty | -110.09 | -110.09 | 0.00 | 68.45 | fuel_cost=-0.013 proximity_reward=0.255 soft_landing_bonus=0.023 velocity_penalty=-0.000 | new_best |
| 9 | fuel_cost + proximity_reward + soft_landing_bonus + velocity_penalty | -110.44 | -110.09 | -0.35 | 68.45 | fuel_cost=-0.014 proximity_reward=0.253 soft_landing_bonus=0.033 velocity_penalty=-0.010 | no_meaningful_improvement |
