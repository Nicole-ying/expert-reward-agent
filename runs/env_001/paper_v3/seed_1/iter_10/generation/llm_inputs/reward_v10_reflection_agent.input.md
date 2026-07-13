# Search objective
- target_score: 200.000000
- current_score: -51.015629
- gap_to_target: 251.015629
- target_achievement_ratio: -25.508%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -51.015629）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters
    w_goal = 1.0
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 3.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Narrow proximity gate for contact reward: only reward leg contact very close to target
    proximity_narrow = 1.0 / (1.0 + 5.0 * dist_sq)

    # Wide proximity gate for velocity penalty: provide deceleration feedback earlier on approach
    proximity_wide = 1.0 / (1.0 + 1.5 * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active over a wider approach zone
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity_wide

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: reward both legs touching the pad, gated by narrow proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity_narrow

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    x = obs[0]            # horizontal position relative to target pad
    y = obs[1]            # vertical position relative to target pad
    vx = obs[2]           # horizontal velocity
    vy = obs[3]           # vertical velocity
    angle = obs[4]        # body angle
    ang_vel = obs[5]      # angular velocity
    left_contact = obs[6] # left support leg contact (0 or 1)
    right_contact = obs[7]# right support leg contact (0 or 1)

    # Hyperparameters (tunable in later iterations)
    w_goal = 1.0
    alpha_proximity = 5.0          # controls the activation radius for proximity-based terms
    w_vel = 0.5
    w_angle = 0.2
    w_angvel = 0.1
    w_contact = 2.0

    # Distance to target center (squared)
    dist_sq = x**2 + y**2

    # Soft proximity weight: ~1 when close to target, ~0 when far
    proximity = 1.0 / (1.0 + alpha_proximity * dist_sq)

    # 1. Main progress: drive toward target center (dense quadratic penalty on distance)
    goal_proximity = -w_goal * dist_sq

    # 2. Soft landing velocity penalty: active only near the target
    velocity_penalty = -w_vel * (vx**2 + vy**2) * proximity

    # 3. Orientation stability penalty: penalize tilt and spin everywhere (light weight)
    orientation_penalty = -w_angle * (angle**2) - w_angvel * (ang_vel**2)

    # 4. Contact reward: encourage both legs touching, gated by proximity
    both_legs_contact = left_contact * right_contact  # 1 only if both are 1
    contact_reward = w_contact * both_legs_contact * proximity

    # Total reward
    total_reward = goal_proximity + velocity_penalty + orientation_penalty + contact_reward

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-51.015629, len=974.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-93.387214, -7.111848]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -258.114002 | -92.2% | 92.2% | 100.0% |
| velocity_penalty | -14.016724 | -5.0% | 5.0% | 100.0% |
| contact_reward | 6.831231 | 2.4% | 2.4% | 0.5% |
| orientation_penalty | -0.884365 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个 2D 飞行器从顶部中央区域安全、快速地到达并稳定停靠在中央目标垫上。  
次目标：尽可能节省引擎推力。  
不应混淆的目标：纯粹的飞行姿态控制不是最终目标；省燃料不能影响成功着陆。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (推测，其中的接触标志为 0.0 或 1.0)  
- 字段含义与可奖励性：

| 索引 | 名称 | 含义 | 可用于奖励？ |
|------|------|------|--------------|
| 0 | x_position | 相对目标垫的水平坐标 | 是 |
| 1 | y_position | 相对目标垫高度的垂直坐标 | 是 |
| 2 | x_velocity | 水平线速度 | 是 |
| 3 | y_velocity | 垂直线速度 | 是 |
| 4 | body_angle | 机体角度 (orientation) | 是 |
| 5 | angular_velocity | 角速度 | 是 |
| 6 | left_support_contact | 左支撑腿接触标志 (0/1) | 是 |
| 7 | right_support_contact | 右支撑腿接触标志 (0/1) | 是 |

所有 8 个量均可在奖励函数中安全使用；全部源自环境本身，无隐藏信息。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：

| 动作 ID | 名称 | 含义 |
|---------|------|------|
| 0 | no_engine | 无任何引擎推力 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生旋转力矩） |
| 2 | main_engine | 启动主引擎（产生主推力，可能同时影响姿态） |
| 3 | right_orientation_engine | 启动右姿态引擎（与左引擎方向相反） |

## 5. step 与终止条件分析
### 5.1 终止模式
- 可能表示成功的终止模式：
  - `body_not_awake_or_settled`：如果同时满足位置接近目标、速度极小、双支撑腿着地且角度稳定，可视为成功着陆。
- 可能表示失败的终止模式：
  - `crash_or_body_contact`：机体非腿部分与障碍物/地面碰撞。
  - `horizontal_position_outside_viewport`：超出水平边界。
- 模糊终止模式：
  - `body_not_awake_or_settled`：在偏离目标或仅有单腿接触时也可能触发，此时可能是失败（例如跌落后停止）。
- 截断：未提及任何时间截断或步骤限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] (info 固定为空字典)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不得使用。终止原因不可从 info 获得。

## 7. 可用于奖励函数的信号
- **位置**：`x_position`, `y_position`（相对目标垫，可直接做距离度量）
- **速度**：`x_velocity`, `y_velocity`（控制着陆时的冲击力与稳定性）
- **姿态**：`body_angle`, `angular_velocity`（稳定竖直）
- **接触**：`left_support_contact`, `right_support_contact`（两条支撑腿是否接触）
- **动作/引擎**：`action`（离散动作索引，可用于燃料消耗度量）
- **其他**：无

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle/lander
  actuator_type: discrete (one main engine, two orientation engines)
  contact_structure: two support legs (binary contacts)
primary_objectives:
  - 到达并停留在目标垫上（位置接近、双支撑腿接触）
  - 低速、稳定着陆（速度小、角度竖直）
secondary_objectives:
  - 尽量少使用引擎推力（节省燃料）
  - 尽快完成着陆（非强制时间最小化）
main_failure_risks:
  - 机体非腿部分与地面碰撞
  - 水平漂出视口
  - 角度过大导致不稳定着陆或接触失败
  - 过早切断引擎导致粗猛着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_reward
  purpose: 驱动机体向目标垫移动并停留在中心
  why_required: 任务核心是到达目标位置，没有此职责 agent 将无方向
  usable_signals: [x_position, y_position]
  risks: 单纯距离奖励可能引导高速冲撞，需配合速度控制

- role_id: soft_landing_velocity_penalty
  purpose: 在接近目标时抑制过大的水平与垂直速度
  why_required: 成功要求“settle”，高速冲击会导致失败或无法稳定
  usable_signals: [x_velocity, y_velocity, x_position, y_position]
  risks: 如果对所有时间惩罚速度，可能阻止探索；应只在靠近目标时激活

- role_id: orientation_stability_reward
  purpose: 保持机体角度稳定（接近 0 或竖直）
  why_required: 着陆需要平稳接触，大角度倾斜可能只接触单腿甚至倾覆
  usable_signals: [body_angle, angular_velocity]
  risks: 角度控制权重过大可能抑制必要的姿态调整；可通过绝对角度惩罚实现

- role_id: contact_reward
  purpose: 奖励双支撑腿同时接触
  why_required: 着陆成功的明确信号；能帮助 agent 理解“settle”状态
  usable_signals: [left_support_contact, right_support_contact]
  risks: 若只奖励接触而不要求位置，agent 可能学会到处触碰；必须与目标位置结合

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  purpose: 减少不必要的引擎使用
  condition_to_use: 当 agent 已经能稳定到达目标附近时，可以逐步引入动作惩罚；或在任意时刻加入极小权重，但确保不压倒主目标
  usable_signals: [action]
  risks: 过早或过大惩罚会使 agent 不敢移动，完全不做探索

- role_id: settlement_bonus
  purpose: 在完全满足着陆条件时给予一次性高奖励
  condition_to_use: 仅在极靠近目标、速度极小、双支撑腿接触且角度稳定的状态下给予；相当于“成功”奖励的代理
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 如果阈值过严可能过于稀疏，难以学习；过松则误触

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_crash_penalty
  reason: 缺少可靠的失败/碰撞标志，终止原因不可从 info 获得；若强行用 next_obs 推断崩溃（例如检测到全零）可能不可靠且引入噪声
  forbidden_or_missing_signals: [collision_flag, termination_reason]

- role_id: time_optimality_penalty
  reason: 任务描述要求“尽快”，但未提供时间步数度量或截断信息；主动针对步数设计奖励可能干扰主要目标，可作为次要考虑，不宜作为强约束
  forbidden_or_missing_signals: [step_count, truncation_flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_reward | x_position, y_position | — | dense_state_signal, euclidean_distance, gaussian_mixture, potential_based | 可用 -distance^2 或 exp(-dist) 等；可配合 shaping |
| soft_landing_velocity_penalty | x_velocity, y_velocity, x_position, y_position | — | quadratic_penalty, masked_by_proximity | 仅在靠近目标时生效，设为平方惩罚 |
| orientation_stability_reward | body_angle, angular_velocity | — | quadratic_penalty, absolute_penalty | 可直接使用 -angle^2, -ang_vel^2 或线性惩罚 |
| contact_reward | left_support_contact, right_support_contact | — | boolean_and, discrete_bonus | 当两个接触标志均为 1 时给予正奖励；与位置耦合 |
| fuel_efficiency_penalty | action | fuel_consumption_per_action (隐含) | action_cost_lookup, scaling | 离散动作：非零动作给予小额惩罚；逐渐引入 |
| settlement_bonus | 全部位置/速度/角度/接触信号 | — | thresholded_bonus | 所有条件逻辑与，成功后给较大常数 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 离目标很远就停止（过早认为任务完成） | 终止时的 x_position, y_position 大，或单腿/无接触 | 加强 goal_proximity_reward，或降低 settlement_bonus 阈值 |
| 高速冲撞目标垫（crash 终止） | 终止前速度大，x,y 速度未衰减，或仅有极短时接触 | 加强 soft_landing_velocity_penalty 或在较远距离就施加减速 |
| 持续悬停不下降（不敢靠近或燃料恐惧） | y_position 缓慢变化，长期不终止，动作多为 0 | 减弱或延迟 fuel_efficiency_penalty，增加目标垫附近的吸引 |
| 角度剧烈摇晃无法稳定 | body_angle 和 angular_velocity 振幅大，接触断续 | 增加 orientation_stability_reward 权重，考虑角度变化率惩罚 |
| 单腿着地反复弹跳 | 仅一侧接触标志为 1，且 x,y 位置略偏 | 提升接触奖励，并对单腿接触给予轻微负奖励（需安全引入） |
| 燃料使用过度，但成功着陆 | 动作序列中大量非零动作 | 在训练后期逐步引入 fuel_efficiency_penalty，进行策略微调 |
| 水平漂出视口 | x_position 绝对值过大触发终止 | 早期强化位置惩罚，或用较大的位置奖励梯度防止漂移 |