# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据当前轮次的训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated判断。

# 工作方式

阅读训练反馈和当前奖励代码，找出最可能导致低分或失败行为的一个组件，修改它。你可以调整系数、替换数学形式、删除组件或添加新组件。修改后输出完整的 `compute_reward` 函数。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件。

# 输出

直接输出完整Python代码。第一个Python code block必须只包含完整且可执行的`compute_reward`函数。
```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -115.832285
- gap_to_target: 315.832285

# 2. Current reward program (score: -115.832285)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next state
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Derived quantities
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    speed_next = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # Weights and constants
    w_approach = 2.0          # shaping reward for closing distance to target
    w_speed = 0.5             # gentle penalty for high speed
    w_angle = 1.0             # penalty for body tilt
    w_angvel = 0.5            # penalty for angular velocity
    survival_penalty = -0.1   # small step penalty to avoid lingering
    main_engine_penalty = -0.3
    side_engine_penalty = -0.03

    # Contact detection
    both_legs_on_platform = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    base_contact_bonus = 100.0 * both_legs_on_platform

    # Soft landing bonus (high reward for gentle, upright touchdown)
    soft_landing_condition = both_legs_on_platform and abs(vy_next) < 0.5 and abs(angle_next) < 0.1
    soft_contact_bonus = 300.0 * soft_landing_condition

    # Shaping: reward for moving closer to the target (potential-based)
    approach_reward = w_approach * (dist_curr - dist_next)

    # State penalties (guiding toward stability and low energy)
    speed_penalty = -w_speed * speed_next
    angle_penalty = -w_angle * (angle_next ** 2)
    angvel_penalty = -w_angvel * (angvel_next ** 2)

    # Engine (fuel) penalty
    engine_penalty = 0.0
    if action == 2:
        engine_penalty = main_engine_penalty
    elif action == 1 or action == 3:
        engine_penalty = side_engine_penalty

    # Total reward
    total_reward = (approach_reward +
                    speed_penalty +
                    angle_penalty +
                    angvel_penalty +
                    survival_penalty +
                    engine_penalty +
                    base_contact_bonus +
                    soft_contact_bonus)

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "survival_penalty": survival_penalty,
        "engine_penalty": engine_penalty,
        "base_contact_bonus": base_contact_bonus,
        "soft_contact_bonus": soft_contact_bonus
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-115.832285, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.820937, -91.955470]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| base_contact_bonus | 120.000000 | 42.7% | 42.7% | 1.8% |
| soft_contact_bonus | 105.000000 | 37.4% | 37.4% | 0.5% |
| speed_penalty | -36.639149 | -13.0% | 13.0% | 100.0% |
| angvel_penalty | -9.765215 | -3.5% | 3.5% | 99.6% |
| survival_penalty | -6.840000 | -2.4% | 2.4% | 100.0% |
| approach_reward | 2.236960 | 0.8% | 0.8% | 100.0% |
| angle_penalty | -0.294270 | -0.1% | 0.1% | 100.0% |
| engine_penalty | -0.108000 | -0.0% | 0.0% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
```
