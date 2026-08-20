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
# 本轮训练反馈 (score=-37.583039)
# Training Feedback

## Final-policy outcome
score=-37.583039, len=947.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-107.997234, 127.780807]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 4.980184 | 34.3% | 49.2% | 100.0% |
| soft_landing | -2.922441 | -20.1% | 20.1% | 100.0% |
| landing_contact_success | 2.312616 | 15.9% | 15.9% | 0.1% |
| orientation_penalty | -2.135488 | -14.7% | 14.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估数据（每10%步数的官方环境得分）
| pct | score_mean | score_std |
|---|---|---|

# 本轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous, speed-dependent)
    w_contact = 5.0
    v_thresh = 0.5
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    if legs_on_ground:
        speed = (next_vx**2 + next_vy**2) ** 0.5
        contact_bonus = w_contact * max(0.0, 1.0 - speed / v_thresh)
    else:
        contact_bonus = 0.0

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```

# 历史 iter_01 训练反馈
# Training Feedback

## Final-policy outcome
score=-120.998945, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-150.975402, -99.021460]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -148.505275 | -51.5% | 51.5% | 100.0% |
| velocity_penalty | -93.225748 | -32.4% | 32.4% | 99.9% |
| angle_penalty | -43.986293 | -15.3% | 15.3% | 100.0% |
| success_proxy | 1.432261 | 0.5% | 0.5% | 0.1% |
| contact_quality | 0.766166 | 0.3% | 0.3% | 0.2% |
| thrust_penalty | -0.215000 | -0.1% | 0.1% | 6.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_01 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    # Encourage small positional errors, low velocities, upright attitude.
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, but only when the body is already
    # near the pad and moving slowly.  This discourages single‑leg touch.
    w_contact = 10.0

    pos_thresh = 0.3            # meters
    vel_thresh = 0.3            # m/s

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    # Both legs must touch (binary product)
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion soft signal)
    # ------------------------------------------------------------------
    # Geometric mean of near‑perfect conditions.  Only active when all
    # signals are close to ideal, providing a final shaping pull.
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15   # radians

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Use geometric mean to avoid product collapse.
    # 1e-12 avoids zero division.
    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 历史 iter_02 训练反馈
# Training Feedback

## Final-policy outcome
score=-183.316556, len=947.350000, terminated=7/20, truncated=13/20, reward_errors=0
score_range=[-284.463711, -99.250079]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_proxy | 5792.231178 | 60.1% | 60.1% | 100.0% |
| goal_proximity | -3768.392629 | -39.1% | 39.1% | 100.0% |
| thrust_penalty | -47.367500 | -0.5% | 0.5% | 100.0% |
| velocity_penalty | -24.149682 | -0.3% | 0.3% | 100.0% |
| angle_penalty | -9.949903 | -0.1% | 0.1% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_02 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity + soft landing  (main learning signal)
    # ------------------------------------------------------------------
    w_prox  = 2.0
    w_vel   = 1.0
    w_ang   = 2.0

    comp_prox  = w_prox * ( -abs(x) - abs(y) )
    comp_vel   = w_vel  * ( -vx**2 - vy**2 )
    comp_angle = w_ang  * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component B – contact quality (stability constraint)
    # ------------------------------------------------------------------
    w_contact = 10.0
    pos_thresh = 0.3
    vel_thresh = 0.3
    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))
    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component C – success proxy (task‑completion continuous signal)
    # ------------------------------------------------------------------
    # Changed from geometric mean of near‑perfect conditions to an additive,
    # continuous reward that activates gradually as each condition improves.
    # This provides a smoother gradient for the agent to follow.
    w_proxy = 2.0   # reduced from 20.0 to avoid overwhelming penalties

    # Distance thresholds that produce a score from 1 (perfect) down to 0
    score_x     = max(0.0, 1.0 - abs(x) / 1.0)
    score_y     = max(0.0, 1.0 - abs(y) / 1.0)
    score_vx    = max(0.0, 1.0 - abs(vx) / 0.5)
    score_vy    = max(0.0, 1.0 - abs(vy) / 0.5)
    score_angle = max(0.0, 1.0 - abs(angle) / 0.3)
    score_contact = (left_c + right_c) * 0.5   # [0,1]

    # Sum of independent scores – provides partial credit
    comp_proxy = w_proxy * (score_x + score_y + score_vx + score_vy +
                            score_angle + score_contact)

    # ------------------------------------------------------------------
    # Component D – thrust penalty (efficiency cost, secondary)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)   # any engine use

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 历史 iter_03 训练反馈
# Training Feedback

## Final-policy outcome
score=-120.253372, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -98.281266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -156.891027 | -53.8% | 53.8% | 100.0% |
| velocity_penalty | -93.439696 | -32.0% | 32.0% | 99.8% |
| angle_penalty | -38.994126 | -13.4% | 13.4% | 100.0% |
| success_proxy | 1.434087 | 0.5% | 0.5% | 0.1% |
| contact_quality | 0.768669 | 0.3% | 0.3% | 0.2% |
| thrust_penalty | -0.182500 | -0.1% | 0.1% | 5.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_03 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity (bounded, soft guidance)
    # ------------------------------------------------------------------
    # Changed from unbounded -abs(x)-abs(y) to a bounded penalty that
    # saturates far from the pad.
    w_prox  = 5.0
    dist_sq = x**2 + y**2
    comp_prox = w_prox * ( - dist_sq / (1.0 + dist_sq) )

    # ------------------------------------------------------------------
    # Component B – velocity penalty
    # ------------------------------------------------------------------
    w_vel = 1.0
    comp_vel = w_vel * ( -vx**2 - vy**2 )

    # ------------------------------------------------------------------
    # Component C – angle penalty
    # ------------------------------------------------------------------
    w_ang = 2.0
    comp_angle = w_ang * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component D – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, gated by position and velocity.
    w_contact = 10.0

    pos_thresh = 0.3
    vel_thresh = 0.3

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component E – success proxy (geometric mean of near‑perfect conditions)
    # ------------------------------------------------------------------
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5

    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component F – thrust penalty (efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 历史 iter_04 训练反馈
# Training Feedback

## Final-policy outcome
score=-1471.587872, len=435.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-2080.965220, -258.626108]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| soft_success | 12716.349643 | 95.7% | 95.7% | 96.2% |
| velocity_penalty | -376.896537 | -2.8% | 2.8% | 100.0% |
| goal_proximity | 163.929425 | 1.2% | 1.2% | 100.0% |
| thrust_penalty | -21.770000 | -0.2% | 0.2% | 100.0% |
| angle_penalty | -3.785140 | -0.0% | 0.0% | 100.0% |
| contact_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_04 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact
    right_c = next_obs[7]   # right leg contact

    # ------------------------------------------------------------------
    # Component A - goal proximity (positive, bounded approach reward)
    # ------------------------------------------------------------------
    w_prox = 2.0
    dist = (x**2 + y**2) ** 0.5
    # bounded reward: 1.0 when dist=0, decays to 0 when dist large
    comp_prox = w_prox * (1.0 - dist / (dist + 1.0))

    # ------------------------------------------------------------------
    # Component B - velocity penalty (soft brake)
    # ------------------------------------------------------------------
    w_vel = 0.5
    comp_vel = w_vel * (-vx**2 - vy**2)

    # ------------------------------------------------------------------
    # Component C - angle penalty
    # ------------------------------------------------------------------
    w_ang = 1.0
    comp_angle = w_ang * (-angle**2 - 0.1 * angvel**2)

    # ------------------------------------------------------------------
    # Component D - contact quality (leg contact gated by distance)
    # ------------------------------------------------------------------
    w_contact = 2.0
    gate = 1.0 / (1.0 + dist)   # near pad -> ~1, far away -> small
    leg_contact = left_c * right_c   # bilateral contact only
    comp_contact = w_contact * leg_contact * gate

    # ------------------------------------------------------------------
    # Component E - soft success shaping (sum of linear scores)
    # ------------------------------------------------------------------
    w_soft = 30.0

    tol_x     = 0.1
    tol_y     = 0.1
    tol_vx    = 0.2
    tol_vy    = 0.2
    tol_angle = 0.2

    score_x     = max(0.0, 1.0 - abs(x) / tol_x)
    score_y     = max(0.0, 1.0 - abs(y) / tol_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / tol_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / tol_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / tol_angle)
    score_leg   = (left_c + right_c) * 0.5   # linear in [0,1]

    soft_sum = (score_x + score_y + score_vx + score_vy +
                score_angle + score_leg)
    comp_soft = w_soft * soft_sum

    # ------------------------------------------------------------------
    # Component F - thrust penalty (fuel efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_soft + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "soft_success": comp_soft,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```

# 历史 iter_05 训练反馈
# Training Feedback

## Final-policy outcome
score=-63.974403, len=963.750000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-95.513496, -20.657539]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 3.779845 | 27.5% | 55.3% | 100.0% |
| soft_landing | -4.930737 | -35.9% | 35.9% | 100.0% |
| orientation_penalty | -1.218466 | -8.9% | 8.9% | 100.0% |
| landing_contact_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 历史 iter_05 奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous, speed-dependent)
    w_contact = 5.0
    v_thresh = 0.5
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    if legs_on_ground:
        speed = (next_vx**2 + next_vy**2) ** 0.5
        contact_bonus = w_contact * max(0.0, 1.0 - speed / v_thresh)
    else:
        contact_bonus = 0.0

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```

# 历史记忆表
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -121.00 | -121.00 | 0.00 | 68.35 | angle_penalty=-0.544 contact_quality=0.021 goal_proximity=-2.156 success_proxy=0.037 thrust_penalty=-0.024 | new_best |
| 2 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -183.32 | -121.00 | -62.32 | 947.35 | angle_penalty=-0.207 contact_quality=0.896 goal_proximity=-1.565 success_proxy=7.092 thrust_penalty=-0.040 | no_meaningful_improvement |
| 3 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -120.25 | -120.25 | 0.00 | 68.40 | angle_penalty=-0.563 contact_quality=0.022 goal_proximity=-2.285 success_proxy=0.037 thrust_penalty=-0.022 | new_best |
| 4 | angle_penalty + contact_quality + goal_proximity + soft_success + thrust_penalty + velocity_penalty | -1471.59 | -120.25 | -1351.33 | 435.40 | angle_penalty=-0.217 contact_quality=0.033 goal_proximity=0.946 soft_success=28.967 thrust_penalty=-0.044 | unsolved_stagnation_fresh_restart |
| 5 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -63.97 | -63.97 | 0.00 | 963.75 | goal_distance_progress=0.013 landing_contact_success=0.915 orientation_penalty=-0.014 soft_landing=-0.021 | new_best |
| 6 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -37.58 | -37.58 | 0.00 | 947.65 | goal_distance_progress=0.013 landing_contact_success=2.123 orientation_penalty=-0.016 soft_landing=-0.023 | new_best |

```
