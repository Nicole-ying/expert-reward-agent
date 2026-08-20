# Duplicate reward retry
The previous generation duplicated iter 2 (runs\env_001\ablation_no_rag_v1\seed_0\iter_02\generation\reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    distance = (x**2 + y**2)**0.5
    proximity_reward = 1.0 / (1.0 + distance)

    tilt_penalty = -0.5 * abs(angle)
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed
    rotation_penalty = -0.1 * abs(angular_vel)

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty
    }
    return total_reward, components
```

# Search objective
- target_score: 200.000000
- current_score: 255.536242
- gap_to_target: -55.536242
- target_achievement_ratio: 127.768%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 255.536242）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    distance = (x**2 + y**2)**0.5
    proximity_reward = 1.0 / (1.0 + distance)

    tilt_penalty = -0.5 * abs(angle)
    speed = (vx**2 + vy**2)**0.5
    velocity_penalty = -0.1 * speed
    rotation_penalty = -0.1 * abs(angular_vel)

    total_reward = proximity_reward + tilt_penalty + velocity_penalty + rotation_penalty

    components = {
        "proximity_reward": proximity_reward,
        "tilt_penalty": tilt_penalty,
        "velocity_penalty": velocity_penalty,
        "rotation_penalty": rotation_penalty
    }
    return total_reward, components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=255.536242, len=345.550000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[172.224289, 292.453207]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 256.183906 | 92.8% | 92.8% | 100.0% |
| tilt_penalty | -10.056788 | -3.6% | 3.6% | 100.0% |
| velocity_penalty | -8.253758 | -3.0% | 3.0% | 99.7% |
| rotation_penalty | -1.450491 | -0.5% | 0.5% | 99.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D类飞行器轨迹优化任务。主体初始位于视口顶部中央附近，并受到随机初始力的作用。智能体的目标是**尽快飞到并稳定停留在中央的目标着陆垫上**，同时尽量少用引擎推力。学习过程中需要实现：逐渐接近目标垫、降低速度、保持稳定的姿态（角度）并安全地完成接触。

## 3. 观察空间 observation_space
- type: Box (连续值)
- shape: (8,)
- dtype: float32 (推断)
- obs[0]: x_position — 相对于目标着陆垫中心的水平坐标
- obs[1]: y_position — 相对于目标着陆垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 主体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右侧支撑接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无任何推力
- action 1: left_orientation_engine — 启动左侧姿态发动机（产生一个方向的角力/力矩）
- action 2: main_engine — 启动主发动机（产生向上的推力，推测）
- action 3: right_orientation_engine — 启动右侧姿态发动机（与左侧相反方向的角力/力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 当主体在目标垫上稳定下来（body_not_awake_or_settled）时，可能被视为成功。该条件检查主体是否已经停止运动并处于睡眠状态，结合位置接近目标垫、速度极低等因素可以判断为成功着陆。
- failure-like termination: 
  - crash_or_body_contact（坠毁或身体触地等异常接触）
  - horizontal_position_outside_viewport（水平坐标超出视口边界）
- ambiguous termination: body_not_awake_or_settled 也可能发生在非目标位置（如坠毁后不再运动），此类终止不一定是成功。
- truncation: 本环境暂未看到基于步数的截断，但通常可能有默认的时间上限（如 1000 步），此类截断不属于成功/失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （info 字典为空，无可用的字段）
- forbidden_or_uncertain_info_fields: 任何试图从 info 中读取 success、failure、termination_reason 等字段的操作均不允许，因为这些字段不存在。

## 7. 可用于奖励函数的信号
- position: obs[0] (x), obs[1] (y) —— 可用于衡量距离目标垫越近越好
- velocity: obs[2] (vx), obs[3] (vy) —— 可用于鼓励终端低速，或在接近目标时减速
- orientation: obs[4] (angle) —— 可用于鼓励保持竖直或特定安全姿态
- angular velocity: obs[5] —— 可惩罚快速旋转
- contact: obs[6], obs[7] —— 两腿是否接触地面，可用于检测着陆及双腿是否平稳
- action/engine: 从动作编号可以判断是否使用了主引擎或姿态发动机，从而度量能量消耗

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | proximity_reward + rotation_penalty + tilt_penalty + velocity_penalty | -110.23 | -110.23 | 0.00 | 68.40 | proximity_reward=-0.973 rotation_penalty=-0.008 tilt_penalty=-0.018 velocity_penalty=-0.106 | new_best |
| 2 | proximity_reward + rotation_penalty + tilt_penalty + velocity_penalty | 255.54 | 255.54 | 0.00 | 345.55 | proximity_reward=0.755 rotation_penalty=-0.007 tilt_penalty=-0.036 velocity_penalty=-0.020 | target_solved_new_best |
