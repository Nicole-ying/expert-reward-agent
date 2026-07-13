# Duplicate reward retry
The previous generation duplicated iter 4 (runs\env_001\paper_v4\seed_3\iter_04\generation\reward_v4.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    total_reward = progress_reward + velocity_penalty + orientation_penalty + landing_reward

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'landing_reward': float(landing_reward)
    }

    return float(total_reward), components
```

# Search objective
- target_score: 200.000000
- current_score: 253.706833
- gap_to_target: -53.706833
- target_achievement_ratio: 126.853%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 253.706833）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function with progress guidance and reachable landing gradient.

    Components:
    - progress: rewards reducing distance to target center (0,0)
    - velocity_penalty: speed penalty gated by proximity to target
    - orientation_penalty: penalty for tilt and angular velocity
    - landing_reward: proximity-gated contact gradient + settling quality bonus
      Replaces the unreachable settling_quality with a staged design:
      (a) contact_reward provides accessible gradient for any leg contact near target
      (b) settling_bonus provides aspirational reward for dual-contact quality landing
    """
    # Current state
    x_pos, y_pos = obs[0], obs[1]
    # Next state
    nx_pos, ny_pos = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distances to target
    prev_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (nx_pos ** 2 + ny_pos ** 2) ** 0.5

    # 1. Progress reward: positive for moving toward target, negative for moving away
    progress_weight = 5.0
    progress_reward = progress_weight * (prev_dist - next_dist)

    # 2. Velocity penalty – active only when close to target
    proximity_factor = 1.0 / (1.0 + 10.0 * next_dist)
    vel_weight = 1.0
    velocity_penalty = -vel_weight * proximity_factor * (nx_vel ** 2 + ny_vel ** 2)

    # 3. Orientation stability – keep body upright and avoid spinning
    orient_weight = 0.5
    orientation_penalty = -orient_weight * (n_angle ** 2 + 0.2 * n_angvel ** 2)

    # 4. Landing reward – staged: accessible contact gradient + quality bonus
    #    contact_score provides continuous gradient: 0 (no legs) / 1 (one leg) / 2 (both legs)
    contact_score = left_contact + right_contact
    contact_weight = 0.3
    contact_reward = contact_weight * proximity_factor * contact_score

    #    settling_bonus: extra reward when both legs contact with low velocity and stable attitude
    vel_magnitude = (nx_vel ** 2 + ny_vel ** 2) ** 0.5
    settling_factor = 1.0 / (1.0 + 5.0 * vel_magnitude + 2.0 * abs(n_angle) + 1.0 * abs(n_angvel))
    contact_product = left_contact * right_contact
    settling_bonus_weight = 0.5
    settling_bonus = settling_bonus_weight * contact_product * proximity_factor * settling_factor

    landing_reward = contact_reward + settling_bonus

    total_reward = progress_reward + velocity_penalty + orientation_penalty + landing_reward

    components = {
        'progress': float(progress_reward),
        'velocity_penalty': float(velocity_penalty),
        'orientation_penalty': float(orientation_penalty),
        'landing_reward': float(landing_reward)
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=253.706833, len=362.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[223.583934, 288.190027]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 35.086909 | 73.9% | 73.9% | 11.3% |
| progress | 6.948097 | 14.6% | 14.9% | 95.9% |
| velocity_penalty | -3.418937 | -7.2% | 7.2% | 96.6% |
| orientation_penalty | -1.901877 | -4.0% | 4.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体是一个二维飞行器，从视口顶部中心附近以随机初始速度开始运动。核心目标：以最小的引擎推力、最短的时间，到达画面中央的目标停靠平台，并在目标位置保持稳定姿态与安全接触。智能体必须学会精确导航、减速软着陆、维持水平姿态并最终使身体稳定（settled）。混淆目标：不能将“尽量少用引擎”误解为完全不使用引擎；不能把“尽快到达”与“粗暴着陆”等价；不能只优化到达而忽略着陆稳定性。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段详解：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 本体相对目标平台的水平坐标 | true |
| 1 | y_position | 本体相对平台高度的垂直坐标（正向可能为上方） | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 本体倾角（弧度） | true |
| 5 | angular_velocity | 角速度（弧度/单位时间） | true |
| 6 | left_support_contact | 左支撑腿/触地点接触标志（1.0接触，0.0离地） | true |
| 7 | right_support_contact | 右支撑腿/触地点接触标志 | true |

所有观测均可用于奖励设计，尤其是位置、速度、角度、接触标志。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作值 | 名称 | 含义 |
|--------|------|------|
| 0 | no_engine | 无推力，依靠当前动量滑行 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生逆时针或顺时针力矩） |
| 2 | main_engine | 启动主引擎（产生垂直向上推力，可能伴随小力矩） |
| 3 | right_orientation_engine | 启动右姿态引擎（产生相反方向力矩） |

动作直接影响本体推力和姿态力矩，间接影响位置、速度、角度、角速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` – 当本体与平台稳定接触且速度/角速度极小，进入物理引擎的“休眠”状态。结合任务目标，若该事件发生在目标平台附近且接触成功，即为成功着陆。但该条件本身仅依赖物理状态，不检查位置是否在目标附近，因此单靠此终止可能包含非目标位置的过早休眠。

- **failure-like termination**:  
  `crash_or_body_contact` – 任何非支撑腿的身体部分接触地面或发生猛烈碰撞，通常表示硬着陆、倾覆或偏离安全姿态。  
  `horizontal_position_outside_viewport` – 水平漂移出画面边界，显然失败。

- **ambiguous termination**: 无。

- **truncation**: step 源码中未显示任何时间步截断，返回 `terminated` 且 `truncated=False`。因此没有时间上限导致的截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: `{}`（step 返回空字典）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为未提供任何信息。

终止条件仅能从 `terminated` 标志判断 episode 结束，无法直接读取 success/failure 标签。任何基于成功/失败的奖励必须从观测状态中自行推断（例如结合位置、接触和休眠）。

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position` (obs[0])，`y_position` (obs[1]) – 用于计算到目标的距离、高度误差。
- **速度信号**：`x_velocity` (obs[2])，`y_velocity` (obs[3]) – 用于鼓励减速、控制着陆冲击。
- **姿态信号**：`body_angle` (obs[4])，`angular_velocity` (obs[5]) – 用于维持水平姿态、稳定角速度。
- **接触信号**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7]) – 用于检测着陆成功、双腿接地、避免单腿/侧倾。
- **动作/引擎信号**：`action` – 用于惩罚引擎使用、鼓励节约。
- **其他组合**：依据连续两帧的 `obs` 与 `next_obs` 可计算位置、速度的变化，评估进展或风险。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity + velocity_penalty | -19.59 | -19.59 | 0.00 | 78.25 | contact_bonus=0.003 orientation_penalty=-0.044 proximity=-2.196 velocity_penalty=-0.211 | new_best |
| 2 | contact_bonus + orientation_penalty + progress + velocity_penalty | 147.71 | 147.71 | 0.00 | 1000.00 | contact_bonus=0.241 orientation_penalty=-0.010 progress=0.013 velocity_penalty=-0.018 | new_best |
| 3 | orientation_penalty + progress + settling_quality + velocity_penalty | -15.42 | 147.71 | -163.13 | 1000.00 | orientation_penalty=-0.012 progress=0.014 settling_quality=0.614 velocity_penalty=-0.020 | no_meaningful_improvement |
| 4 | landing_reward + orientation_penalty + progress + velocity_penalty | 253.71 | 253.71 | 0.00 | 362.10 | landing_reward=0.407 orientation_penalty=-0.012 progress=0.015 velocity_penalty=-0.023 | target_solved_new_best |
