# 1. Search objective
- target_score: 200.000000
- current_score: 58.862720
- gap_to_target: 141.137280

# 2. Current reward program (score: 58.862720)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations for current and next state
    # obs indices:
    # 0: x_position, 1: y_position, 2: x_velocity, 3: y_velocity,
    # 4: body_angle, 5: angular_velocity, 6: left_contact, 7: right_contact
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Compute distances from target (0,0)
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights (heuristically reasonable starting points for this environment)
    w_progress = 1.0           # scale for distance improvement
    w_speed   = 0.5            # penalty for residual kinetic energy
    w_angle   = 0.5            # penalty for tilt
    w_angvel  = 0.5            # penalty for rotation
    w_contact = 10.0           # terminal soft‑landing incentive
    alpha     = 1.0            # sharpness of vertical‑speed gate
    beta      = 1.0            # sharpness of tilt gate

    # 1. Main progress signal: approach + velocity damping
    #    Use improvement_delta for distance (reward getting closer)
    #    and penalize high speed in next state.
    progress_reward = w_progress * (dist_curr - dist_next) - w_speed * speed_next

    # 2. Orientation stabilization (safety constraint)
    orientation_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # 3. Soft‑landing contact proxy (approximate task completion)
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    # Use exponential decay to favour landing with low vertical speed and near‑vertical attitude.
    # exp(-alpha*|vy|) is computed as e ** (‑alpha * abs(vy))
    # exp(‑beta * angle²) is computed similarly.
    smooth_vy_gate = 2.718281828 ** (-alpha * abs(vy_next))
    smooth_angle_gate = 2.718281828 ** (-beta * (angle_next ** 2))
    soft_contact_reward = w_contact * both_legs_on_platform * smooth_vy_gate * smooth_angle_gate

    total_reward = progress_reward + orientation_penalty + soft_contact_reward

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_contact_reward": soft_contact_reward
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=58.862720, len=559.900000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-211.881499, 215.423021]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_contact_reward | 292.891768 | 84.2% | 84.2% | 5.6% |
| progress_reward | -50.796714 | -14.6% | 14.6% | 99.9% |
| orientation_penalty | -3.967178 | -1.1% | 1.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
本环境是一个二维飞行器/着陆器轨迹优化问题。  
智能体从靠近画面顶部中心的位置出发，受到一个随机初始力影响。  
**主要目标**：尽可能快地到达并平稳降落在中央的目标平台上（位置接近目标、速度趋近于零、姿态稳定、双支撑腿安全接触）。  
**次要目标**：最小化发动机推力使用，从而节省燃料。  
**不应混淆的目标**：不要求复杂机动或避障，核心是精准、节能的最终软着陆。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32 (推断，通常连续观测用浮点数)
- 各维度含义（均为 reward usable）：
  - obs[0] **x_position**: 相对目标平台中心的水平坐标 → `reward_usable: true`
  - obs[1] **y_position**: 相对平台高度的垂直坐标 → `reward_usable: true`
  - obs[2] **x_velocity**: 水平线速度 → `reward_usable: true`
  - obs[3] **y_velocity**: 垂直线速度 → `reward_usable: true`
  - obs[4] **body_angle**: 机体朝向角度 → `reward_usable: true`
  - obs[5] **angular_velocity**: 角速度 → `reward_usable: true`
  - obs[6] **left_support_contact**: 左支撑腿接触标志（0/1） → `reward_usable: true`
  - obs[7] **right_support_contact**: 右支撑腿接触标志（0/1） → `reward_usable: true`

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- 各动作含义：
  - action 0: **no_engine** —— 不点火（无推力）
  - action 1: **left_orientation_engine** —— 点燃左侧姿态发动机（产生旋转力矩，推左）
  - action 2: **main_engine** —— 点燃主发动机（产生向上推力）
  - action 3: **right_orientation_engine** —— 点燃右侧姿态发动机（产生相反旋转力矩，推右）

## 5. step 与终止条件分析
### 5.1 终止模式
根据源码中 `terminated` 的逻辑：
- **success-like termination**:  
  - `body_not_awake_or_settled`（机体不再活跃或已稳定下来）——在目标平台稳定着陆时通常触发，可视为潜在成功终止。  
  - `crash_or_body_contact` 中的一部分：如果两条腿都与平台接触且速度、角度满足安全条件，可能触发终止并成功，但任务源码未区分成功/失败，故不能直接当作成功标志。
- **failure-like termination**:  
  - `horizontal_position_outside_viewport`（水平位置超出画面边界）——明确失败。  
  - `crash_or_body_contact` 中的非平台接触（例如撞地、侧翻）——明确失败。
- **ambiguous termination**:  
  - `body_not_awake_or_settled` 也可能在失败状态（如翻转昏迷）出现，因此单独依赖此条件不可靠。
- **truncation**:  
  - 源码中未出现 `truncated`，仅 `terminated` 被返回，`info` 为空，因此无其他截断信息。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  `info` 为空，`terminated` 本身也未分解为成功/失败。
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 无（info 固定为 `{}`）
- **forbidden_or_uncertain_info_fields**: 所有未声明的 info 字段均不可用；尤其 **不能假设存在 `success`、`failure`、`termination_reason` 等字段**。

## 7. 可用于奖励函数的信号
以下信号均从 `next_obs` 获取，部分可利用 `obs` 进行 delta 计算（如速度变化）：

- **位置信号**:
  - `next_obs[0]` (x 距目标)
  - `next_obs[1]` (y 距平台)
- **速度信号**:
  - `next_obs[2]` (vx)
  - `next_obs[3]` (vy)
- **姿态信号**:
  - `next_obs[4]` (角度)
  - `next_obs[5]` (角速度)
- **接触信号**:
  - `next_obs[6]` (左腿接触)
  - `next_obs[7]` (右腿接触)
- **动作/推力信号**:
  - `action` (可判断是否使用主发动机或姿态发动机，用于推力惩罚)
- **其他可能衍生信号**:
  - 综合距离 `sqrt(x_pos^2 + y_pos^2)`
  - 速率 `sqrt(vx^2 + vy^2)`
  - 角度绝对值 `abs(angle)`