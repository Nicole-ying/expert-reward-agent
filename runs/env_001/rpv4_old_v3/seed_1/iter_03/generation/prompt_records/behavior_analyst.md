# Prompt Record

## System Prompt

```text
你是训练数据记者。你的唯一职责是：把训练反馈和迭代历史整理成客观的数据报告。你**不做诊断、不给建议、不推荐修改任何组件**。

# 输入数据

你有全部迭代的完整信息，不只是当前轮：

1. **本轮训练反馈** + **本轮 reward 代码**
2. **全部历史轮次的训练反馈 + reward 代码**（用于构建骨架演进表）
3. **历史记忆表**（每轮 skeleton、score、len、action）
4. **Checkpoint 评估数据**（每 10% 步数的官方得分轨迹）

# 输出格式

## 1. 行为事实
[用客观数字描述 agent 的行为，不做因果推断]
- 得分：score=X，range=[min, max]
- 终止方式：terminated X/20（成功/失败），truncated Y/20（时间上限）
- 回合长度：len=Z
- Checkpoint 轨迹：官方得分从 cp1=X1 → cp10=X10（上升/下降/持平）

## 2. 骨架演进表
```
| iter | score | len | term | compA_ep_sum | compA_active | compB_ep_sum | compB_active | ... |
| N-2  |       |     |      |              |              |              |              |     |
| N-1  |       |     |      |              |              |              |              |     |
| N    |       |     |      |              |              |              |              |     |
```

## 3. 数字标记
仅标记以下客观事实，不解释原因，不给建议：
- **主导**：某组件 magnitude_share > 80%
- **死组件**：某组件 active_rate < 2%
- **淹没**：某组件 active_rate > 90% 但 magnitude_share < 3%
- **全覆盖惩罚**：某组件 active_rate > 95% 且 signed_share 为负
- **量级对比**：列出每步均值（ep_sum ÷ len）最大的两个组件和最小的两个组件
- **修改变化**：如果存在上一轮，列出每个组件 ep_sum 和 active_rate 的 Δ

```

## User Prompt

```markdown
# 本轮训练反馈 (score=-109.576065)
# Training Feedback

## Final-policy outcome
score=-109.576065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.894147, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.091045 | -94.3% | 94.3% | 100.0% |
| goal_approach | 1.120194 | 5.5% | 5.7% | 100.0% |
| crash_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估数据（每10%步数的官方环境得分）
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    #    Encourage low speed, upright posture, and slow angular motion.
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. crash_avoidance: unchanged (penalise hard crash below pad without both legs)
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-214.161811, len=182.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-254.447143, -175.657869]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_approach | 0.152428 | 44.8% | 100.0% | 100.0% |
| crash_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |
| soft_landing | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 9/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for goal_approach_and_soft_contact (discrete thrusters, 2D lander).
    """
    # ------------------------------------------------------------------
    # unpack observations (current)
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    # unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ------------------------------------------------------------------
    # 1. goal_approach (potential‑based shaping using L2 distance to pad)
    #    target is (0,0) on the pad.
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # ------------------------------------------------------------------
    # 2. soft_landing – conditional velocity/angular‑velocity penalty near ground
    #    gate rises as altitude drops below threshold
    LANDING_THRESHOLD = 0.5           # meters above pad
    land_weight = max(0.0, 1.0 - (y_pos_next / LANDING_THRESHOLD))
    # quadratic penalty on linear + angular speed in next state
    soft_landing = -land_weight * (
        x_vel_next ** 2 + y_vel_next ** 2 + 0.1 * ang_vel_next ** 2
    )

    # ------------------------------------------------------------------
    # 3. crash_avoidance – harsh penalty when a crash‑like state is reached
    #    crash = below pad (negative height) AND not both legs on contact
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    # ------------------------------------------------------------------
    # Combine components
    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=-109.576065, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.894147, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.091045 | -94.3% | 94.3% | 100.0% |
| goal_approach | 1.120194 | 5.5% | 5.7% | 100.0% |
| crash_avoidance | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    #    Encourage low speed, upright posture, and slow angular motion.
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. crash_avoidance: unchanged (penalise hard crash below pad without both legs)
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |

```
