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

## 4. 语义演化（跨轮组件功能追踪）

按**语义功能**对组件分组，追踪同一功能在多个轮次中的变化。不要用组件名称分组——组件可能在迭代中被重命名，但功能相同。

### 如何识别语义功能

根据组件的**数学形态和奖励方向**判断其功能角色：
- **接近/进度信号**：基于位置/距离的引导，通常为正或 delta 形式（如 goal_approach, progress, proximity, approach_shaping, landing_proximity, goal_distance_progress）
- **着陆/接触信号**：基于腿部接触或着陆事件的奖励/惩罚（如 crash_avoidance, landing_success, contact_reward, touchdown_bonus, stable_landed, safe_contact, contact_bonus, settling_quality, landing_reward）
- **稳定性/约束信号**：基于速度/角度/角速度/推力的惩罚或约束（如 soft_landing, orientation_penalty, velocity_penalty, stability_penalty, fuel_penalty, descent_safety, engine_penalty）
- **完成/成功代理**：条件性的大额正信号或乘积形式的完成接近度（如 success_proxy, completion_bonus）

### 输出格式

```
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆信号 | v1_crash_avoidance → v3_landing_success → v4_contact_reward → v5_landing_success | -214 → -115 → -111 → ? | 稀疏bonus↔连续奖励：2次切换 |
| 稳定性 | v1_soft_landing(死) → v2_soft_landing(全覆盖惩罚) | -214 → -110 | 门控→全高度 |
```

**方向切换**列记录同一语义功能在"稀疏bonus/连续奖励"、"状态值/改善量"、"门控/全高度"等对立设计之间的切换次数。切换 ≥ 2 次意味着反思在该功能上可能正在振荡。只报告客观事实，不给出"应该怎么做"的建议。

```

## User Prompt

```markdown
# 本轮训练反馈 (score=255.276942)
# Training Feedback

## Final-policy outcome
score=255.276942, len=351.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[164.967665, 299.707178]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 25.830000 | 96.4% | 96.4% | 36.7% |
| progress | 0.843410 | 3.1% | 3.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
    # 当前观测
    x, y = obs[0], obs[1]
    # 下一观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = abs(next_obs[4])          # body_angle
    nangvel = abs(next_obs[5])         # angular_velocity
    nleft = next_obs[6]
    nright = next_obs[7]

    # 主进展信号：与目标着陆垫的距离变化
    d_current = (x**2 + y**2)**0.5
    d_next = (nx**2 + ny**2)**0.5 + 1e-8   # 避免除零
    progress = d_current - d_next          # >0 表示接近目标

    # soft health gate —— 在姿态、速度、角速度恶化时衰减正向进展
    # 使用指数衰减，温度系数根据任务特性设定
    angle_gate   = 2.718281828**(-2.0 * nangle**2)                       # 身体倾角过大时衰减
    v_sq = nvx**2 + nvy**2
    vel_gate     = 2.718281828**(-1.0 * v_sq / (1.0 + d_next))           # 高速时衰减，距离越近越严格
    angvel_gate  = 2.718281828**(-3.0 * nangvel)                         # 角速度过大时衰减

    gate = angle_gate * vel_gate * angvel_gate

    # 仅对正向进展施加 gate，负进展（远离）保留完整惩罚
    if progress > 0.0:
        progress *= gate

    # 任务完成近似信号：双腿同时接触着陆垫时给予持续性小奖励
    contact_reward = 0.2 if (nleft > 0.5 and nright > 0.5) else 0.0

    total_reward = progress + contact_reward
    components = {
        "progress": progress,
        "contact": contact_reward
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

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=-115.206637, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.247427, -100.640778]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success | 25.000000 | 53.3% | 53.3% | 0.4% |
| soft_landing | -20.743517 | -44.2% | 44.2% | 100.0% |
| goal_approach | 1.118841 | 2.4% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
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
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_success: sparse positive reward for safe touchdown
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    STABLE = (abs(x_vel_next) < 0.5 and abs(y_vel_next) < 0.5 and 
              abs(body_angle_next) < 0.2 and abs(ang_vel_next) < 0.2)
    if BOTH_LEGS and STABLE:
        landing_success = 100.0
    else:
        landing_success = 0.0

    total_reward = goal_approach + soft_landing + landing_success

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_success": landing_success,
    }

    return float(total_reward), components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=-111.061357, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.746822, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.296874 | -89.3% | 89.3% | 100.0% |
| goal_approach | 1.120424 | 5.2% | 5.4% | 100.0% |
| contact_reward | 1.150000 | 5.3% | 5.3% | 1.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
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
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. contact_reward: continuous per-step reward for having both legs on pad
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    if BOTH_LEGS:
        contact_reward = 1.0
    else:
        contact_reward = 0.0

    total_reward = goal_approach + soft_landing + contact_reward

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-119.003231, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -101.141368]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success | 25.000000 | 53.4% | 53.4% | 0.4% |
| soft_landing | -20.640502 | -44.1% | 44.1% | 100.0% |
| goal_approach | 1.116932 | 2.4% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_05 奖励函数代码
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
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_success: one-time bonus for stable two-leg touchdown
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    STABLE = (abs(x_vel_next) < 1.0 and abs(y_vel_next) < 1.0 and
              abs(body_angle_next) < 0.5 and abs(ang_vel_next) < 1.0)
    if BOTH_LEGS and STABLE:
        landing_success = 100.0
    else:
        landing_success = 0.0

    total_reward = goal_approach + soft_landing + landing_success

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_success": landing_success,
    }

    return float(total_reward), components
```

# 历史 iter_06 训练反馈
# Training Feedback

## Final-policy outcome
score=-108.177815, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.756541, -91.564287]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_landing | -19.127200 | -92.3% | 92.3% | 100.0% |
| goal_approach | 1.119989 | 5.4% | 5.6% | 100.0% |
| landing_comfort | 0.428127 | 2.1% | 2.1% | 0.7% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_06 奖励函数代码
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
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. landing_comfort: continuous reward when both legs touch, scaled by stability
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    if BOTH_LEGS:
        # comfort decreases with velocity and orientation errors
        stability_error = vel_penalty + angle_penalty + ang_vel_penalty
        comfort = max(0.0, 1.0 - stability_error)
        landing_comfort = comfort  # max +1.0 per step when perfectly stable
    else:
        landing_comfort = 0.0

    total_reward = goal_approach + soft_landing + landing_comfort

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "landing_comfort": landing_comfort,
    }

    return float(total_reward), components
```

# 修改自传（反思LLM过去每轮的修改决策和结果）
# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing` → 分数 -214.2 → -109.6（+104.6）✅ 大幅改善
- **你做了什么**：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
- **你为什么这么做**：加入全程速度/姿态惩罚后，agent 将学会使用主引擎抵消下沉、利用姿态引擎抑制翻滚，从而降低惩罚，并与 goal_approach 共同引导向中心垫缓慢降落，最终提高 score 并可能产生腿接触。
- **回合长度**：len: 182→68

### v3：修改 `crash_avoidance` → 分数 -109.6 → -115.2（-5.6）➖ 小幅退步
- **你做了什么**：唯一目标组件为 `crash_avoidance`，将其替换为 `landing_success`：当双腿接触且满足低速、直立、低角速度条件时，给予一次性正向大奖励。
- **你为什么这么做**：增加 100.0 的着陆成功奖励后，agent 将在 soft_landing 的引导下逐步探索到安全着陆状态，一旦成功即获得巨大正奖励，从而使平均分数由负转正；下一轮将观察到 landing_success 的 active_rate > 0 且 episode_sum_mean > 0，scor...
- **回合长度**：len: 68→68

### v4：修改 `landing_success` → 分数 -115.2 → -111.1（+4.1）➖ 小幅改善
- **你做了什么**：唯一目标组件为 landing_success，将其替换为 contact_reward（每步当 left_contact_next>0.5 且 right_contact_next>0.5 时给予 +1.0），并移除 crash_avoi...
- **你为什么这么做**：连续接触奖励会为 agent 提供直接梯度，促使其尽快放下双腿接触垫面，同时 soft_landing 仍会惩罚高速/倾斜，从而引导它学会在接触后继续减速并保持稳定，最终提高成功着陆率并拉升 score。
- **回合长度**：len: 68→68

### v5：修改 `contact_reward` → 分数 -111.1 → -119.0（-7.9）➖ 小幅退步
- **你做了什么**：唯一目标组件contact_reward，将其替换为landing_success，当同时满足双腿接触、低速、直立、低角速度条件时给予一次性+100奖励。
- **你为什么这么做**：一旦landing_success可被触发，代理将获得巨大成正奖励，远超soft_landing负惩罚，从而学会在接近垫面时主动减速并维持姿态，进而提高成功着陆率并拉升score。
- **回合长度**：len: 68→68

### v6：修改 `landing_success` → 分数 -119.0 → -108.2（+10.8）➖ 小幅改善
- **你做了什么**：基于v2代码，把crash_avoidance（几乎不触发）替换为landing_comfort，该组件在双腿接触时根据速度和姿态误差给出连续奖励，最大每步1.0，从而提供逐步改善的梯度。
- **你为什么这么做**：加入连续舒适奖励后，agent将在双腿接触后为获取更高奖励而主动降低速度、减小倾斜和角速度，从而提升成功着陆率，使score上升并突破-109.58。
- **回合长度**：len: 68→68

## 你的修改轨迹（按组件分组）

- **`landing_success`**：修改了 **2 次**，分数变化：v4(+4) → v6(+11)
- **`soft_landing`**：修改了 **1 次**，分数变化：v2(+105)
- **`crash_avoidance`**：修改了 **1 次**，分数变化：v3(-6)
- **`contact_reward`**：修改了 **1 次**，分数变化：v5(-8)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |
| 3 | goal_approach + landing_success + soft_landing | -115.21 | -109.58 | -5.63 | 68.35 | goal_approach=0.016 landing_success=0.536 soft_landing=-0.290 | no_meaningful_improvement |
| 4 | contact_reward + goal_approach + soft_landing | -111.06 | -109.58 | -1.49 | 68.40 | contact_reward=0.017 goal_approach=0.016 soft_landing=-0.281 | no_meaningful_improvement |
| 5 | goal_approach + landing_success + soft_landing | -119.00 | -109.58 | -9.43 | 68.40 | goal_approach=0.016 landing_success=0.643 soft_landing=-0.288 | unsolved_stagnation_fresh_restart |
| 6 | goal_approach + landing_comfort + soft_landing | -108.18 | -108.18 | 0.00 | 68.45 | goal_approach=0.016 landing_comfort=0.006 soft_landing=-0.280 | unsolved_stagnation_fresh_restart |
| 7 | contact + progress | 255.28 | 255.28 | 0.00 | 351.65 | contact=0.111 progress=0.001 | target_solved_new_best |

```
