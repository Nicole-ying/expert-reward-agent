# Duplicate reward retry
The previous generation duplicated iter 7 (runs/env_001/rpv4_old_v2/seed_2/iter_07/generation/reward_v7.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
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

    # 2. Soft landing (proximity-gated velocity penalty) -- restored iter 7 rational gate
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous proximity-gated quality)
    w_success = 5.0
    k_dist_success = 5.0
    k_speed = 2.0
    k_angle_success = 5.0
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    contact_flag = 1.0 if legs_on_ground else 0.0
    dist = (next_x**2 + next_y**2) ** 0.5
    speed = (next_vx**2 + next_vy**2) ** 0.5
    angle_abs = abs(next_angle)
    quality = (2.718281828 ** (-k_dist_success * dist)) * \
              (2.718281828 ** (-k_speed * speed)) * \
              (2.718281828 ** (-k_angle_success * angle_abs))
    contact_bonus = w_success * contact_flag * quality

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

# 1. Search objective
- target_score: 200.000000
- current_score: -12.569633
- gap_to_target: 212.569633
- target_achievement_ratio: -6.285%

# 2. 上一轮奖励函数代码（该轮得分: -12.569633）
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

    # 2. Soft landing (exponential proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 10.0
    proximity_gate = 2.718281828 ** (-k_dist * dist_next)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous proximity-gated quality)
    w_success = 5.0
    k_dist_success = 5.0
    k_speed = 2.0
    k_angle_success = 5.0
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    contact_flag = 1.0 if legs_on_ground else 0.0
    dist = (next_x**2 + next_y**2) ** 0.5
    speed = (next_vx**2 + next_vy**2) ** 0.5
    angle_abs = abs(next_angle)
    quality = (2.718281828 ** (-k_dist_success * dist)) * \
              (2.718281828 ** (-k_speed * speed)) * \
              (2.718281828 ** (-k_angle_success * angle_abs))
    contact_bonus = w_success * contact_flag * quality

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

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 68.35 | -121.00 | ✅ |
| 2 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 947.35 | -183.32 | ❌ |
| 3 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 68.40 | -120.25 | ✅ |
| 4 | 骨架变化: angle_penalty + contact_quality + goal_proximity + | — | 435.40 | -1471.59 | ❓ |
| 5 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 963.75 | -63.97 | ✅ |
| 6 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 947.65 | -37.58 | ✅ |
| 7 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 577.95 | 156.75 | ✅ |
| 8 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 1000.00 | -18.99 | ❌ |
| 9 | 骨架变化: goal_distance_progress + landing_contact_success + | — | 1000.00 | -12.57 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 组件演化

## 基线 iter_01 (score=-121.0, len=68)
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

## iter_02 (score=-183.3, len=947)
- **angle_penalty**: `comp_angle = w_ang  * ( -angle**2 - angvel**2 )`  [未变]
- **contact_quality**: `comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy`  [未变]
- **goal_distance_progress**: `(已删除)`  [未变]
- **goal_proximity**: `(var=comp_prox)`  [未变]
- **landing_contact_success**: `(已删除)`  [未变]
- **orientation_penalty**: `(已删除)`  [未变]
- **soft_landing**: `(已删除)`  [未变]
- **soft_success**: `(已删除)`  [未变]
- **success_proxy**: `comp_proxy = w_proxy * (score_x + score_y + score_vx + score_vy +`  [修改]
- **thrust_penalty**: `comp_thrust = -w_thrust * float(action != 0)   # any engine use`  [未变]
- **velocity_penalty**: `(var=comp_vel)`  [未变]

## iter_03 (score=-120.3, len=68)
- **angle_penalty**: `comp_angle = w_ang * ( -angle**2 - angvel**2 )`  [修改]
- **contact_quality**: `comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy`  [未变]
- **goal_distance_progress**: `(已删除)`  [未变]
- **goal_proximity**: `comp_prox = w_prox * ( - dist_sq / (1.0 + dist_sq) )`  [修改]
- **landing_contact_success**: `(已删除)`  [未变]
- **orientation_penalty**: `(已删除)`  [未变]
- **soft_landing**: `(已删除)`  [未变]
- **soft_success**: `(已删除)`  [未变]
- **success_proxy**: `comp_proxy = 0.0`  [修改]
- **thrust_penalty**: `comp_thrust = -w_thrust * float(action != 0)`  [修改]
- **velocity_penalty**: `comp_vel = w_vel * ( -vx**2 - vy**2 )`  [修改]

## iter_04 (score=-1471.6, len=435) [RESTART]
- **angle_penalty**: `comp_angle = w_ang * (-angle**2 - 0.1 * angvel**2)`  [修改]
- **contact_quality**: `comp_contact = w_contact * leg_contact * gate`  [修改]
- **goal_distance_progress**: `(已删除)`  [未变]
- **goal_proximity**: `comp_prox = w_prox * (1.0 - dist / (dist + 1.0))`  [修改]
- **landing_contact_success**: `(已删除)`  [未变]
- **orientation_penalty**: `(已删除)`  [未变]
- **soft_landing**: `(已删除)`  [未变]
- **soft_success**: `comp_soft = w_soft * soft_sum`  [修改]
- **success_proxy**: `(已删除)`  [修改]
- **thrust_penalty**: `comp_thrust = -w_thrust * float(action != 0)`  [未变]
- **velocity_penalty**: `comp_vel = w_vel * (-vx**2 - vy**2)`  [修改]

## iter_06 (score=-37.6, len=948) [RESTART]
- **angle_penalty**: `(已删除)`  [修改]
- **contact_quality**: `(已删除)`  [修改]
- **goal_distance_progress**: `progress = w_progress * (dist_prev - dist_next)`  [修改]
- **goal_proximity**: `(已删除)`  [修改]
- **landing_contact_success**: `contact_bonus = 0.0`  [修改]
- **orientation_penalty**: `angle_penalty = -w_angle * (next_angle**2)`  [修改]
- **soft_landing**: `vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate`  [修改]
- **soft_success**: `(已删除)`  [修改]
- **success_proxy**: `(已删除)`  [未变]
- **thrust_penalty**: `(已删除)`  [修改]
- **velocity_penalty**: `(已删除)`  [修改]

## iter_07 (score=156.7, len=578) [BEST]
- **angle_penalty**: `(已删除)`  [未变]
- **contact_quality**: `(已删除)`  [未变]
- **goal_distance_progress**: `progress = w_progress * (dist_prev - dist_next)`  [未变]
- **goal_proximity**: `(已删除)`  [未变]
- **landing_contact_success**: `contact_bonus = w_success * contact_flag * quality`  [修改]
- **orientation_penalty**: `angle_penalty = -w_angle * (next_angle**2)`  [未变]
- **soft_landing**: `vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate`  [未变]
- **soft_success**: `(已删除)`  [未变]
- **success_proxy**: `(已删除)`  [未变]
- **thrust_penalty**: `(已删除)`  [未变]
- **velocity_penalty**: `(已删除)`  [未变]

## iter_08 (score=-19.0, len=1000)
- **angle_penalty**: `(已删除)`  [未变]
- **contact_quality**: `(已删除)`  [未变]
- **goal_distance_progress**: `progress = w_progress * (dist_prev - dist_next)`  [未变]
- **goal_proximity**: `(已删除)`  [未变]
- **landing_contact_success**: `contact_bonus = w_success * contact_flag * quality`  [未变]
- **orientation_penalty**: `angle_penalty = -w_angle * (next_angle**2)`  [未变]
- **soft_landing**: `vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate`  [未变]
- **soft_success**: `(已删除)`  [未变]
- **success_proxy**: `(已删除)`  [未变]
- **thrust_penalty**: `(已删除)`  [未变]
- **velocity_penalty**: `(已删除)`  [未变]

## iter_09 (score=-12.6, len=1000)
- **angle_penalty**: `(已删除)`  [未变]
- **contact_quality**: `(已删除)`  [未变]
- **goal_distance_progress**: `progress = w_progress * (dist_prev - dist_next)`  [未变]
- **goal_proximity**: `(已删除)`  [未变]
- **landing_contact_success**: `contact_bonus = w_success * contact_flag * quality`  [未变]
- **orientation_penalty**: `angle_penalty = -w_angle * (next_angle**2)`  [未变]
- **soft_landing**: `vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate`  [未变]
- **soft_success**: `(已删除)`  [未变]
- **success_proxy**: `(已删除)`  [未变]
- **thrust_penalty**: `(已删除)`  [未变]
- **velocity_penalty**: `(已删除)`  [未变]


# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
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

    # 4. Landing contact success (continuous proximity-gated quality)
    w_success = 5.0
    k_dist_success = 5.0
    k_speed = 2.0
    k_angle_success = 5.0
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    contact_flag = 1.0 if legs_on_ground else 0.0
    dist = (next_x**2 + next_y**2) ** 0.5
    speed = (next_vx**2 + next_vy**2) ** 0.5
    angle_abs = abs(next_angle)
    quality = (2.718281828 ** (-k_dist_success * dist)) * \
              (2.718281828 ** (-k_speed * speed)) * \
              (2.718281828 ** (-k_angle_success * angle_abs))
    contact_bonus = w_success * contact_flag * quality

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

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| goal_distance_progress | 0.012 | 0.015 | +0.003 | 99.9% | 99.9% | -0.1pp |
| landing_contact_success | 1.333 | 1.505 | +0.171 | 49.6% | 54.8% | +5.1pp |
| orientation_penalty | -0.016 | -0.010 | +0.006 | 100.0% | 100.0% | -0.0pp |
| soft_landing | -0.038 | -0.004 | +0.034 | 100.0% | 99.9% | -0.1pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -18.99 | -12.57 | +6.42 |
| 平均回合长度 | 1000.0 | 1000.0 | +0.0 |
| 终止回合数 | 0/20 | 0/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-12.569633, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-43.381619, 24.541211]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_distance_progress | 6.281804 | 65.1% | 76.8% | 100.0% |
| orientation_penalty | -1.594844 | -16.5% | 16.5% | 100.0% |
| soft_landing | -0.643557 | -6.7% | 6.7% | 100.0% |
| landing_contact_success | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主体从视野顶部中央附近以随机初始速度出发，目标是尽可能快速、稳定地降落在中央目标垫上并停稳（双腿接触、位置居中对齐、姿态水平、速度几乎为零）。次级目标是整个过程中尽量少使用发动机推力（省燃料）。不应混淆的目标：单纯的“存活”或“保持平衡”不是主目标，而是着陆成功的前提；单纯的“燃料为零”也不是主目标，只是在成功着陆前提下才受鼓励。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64 或 float32（由环境决定）  
- obs[0]: name=x_position, meaning=相对于目标垫水平位置的偏差坐标，reward_usable: true  
- obs[1]: name=y_position, meaning=相对于目标垫高度的垂直偏差坐标，reward_usable: true  
- obs[2]: name=x_velocity, meaning=水平线速度，reward_usable: true  
- obs[3]: name=y_velocity, meaning=垂直线速度，reward_usable: true  
- obs[4]: name=body_angle, meaning=机体倾斜角度（弧度制），reward_usable: true  
- obs[5]: name=angular_velocity, meaning=角速度，reward_usable: true  
- obs[6]: name=left_support_contact, meaning=左支撑腿接触标志（1.0 表示接触），reward_usable: true  
- obs[7]: name=right_support_contact, meaning=右支撑腿接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: name=no_engine, meaning=不点火，无推力  
- action 1: name=left_orientation_engine, meaning=点燃左姿态发动机（产生一个力矩和微小横向力）  
- action 2: name=main_engine, meaning=点燃主发动机（产生向上推力）  
- action 3: name=right_orientation_engine, meaning=点燃右姿态发动机（产生反向力矩和微小横向力）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled：机体进入睡眠或完全稳定不动。当双腿安全接触目标垫且速度、姿态均在阈值内时可能触发，属于成功信号（但无法直接与失败情形区分）。  
- failure-like termination:  
  - crash_or_body_contact：机体与地面或障碍猛烈碰撞（可能由于姿态过度倾斜或高速冲击非垫区域）→ 典型失败。  
  - horizontal_position_outside_viewport：水平位置超出有效视野范围 → 失败。  
- ambiguous termination:  
  - body_not_awake_or_settled 若发生在垫外（比如在倾斜坡道上稳定）也可能被终止，此时实际是失败，但终止模式相同。  
- truncation: 未在代码片段中明确给出最大步数，但通常存在隐含截断。本分析不依赖截断信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: 无。info 恒为空字典，不可用任何 info 键。  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用。

## 7. 可用于奖励函数的信号
- position: obs[0] (x 偏差), obs[1] (y 偏差) → 目标接近程度
- velocity: obs[2] (vx), obs[3] (vy) → 移动速度，用于鼓励减速
- orientation: obs[4] (body_angle) → 倾斜程度，应接近0
- angular_velocity: obs[5] → 角旋转速率，可辅助惩罚快速旋转
- contact: obs[6] (左腿接触), obs[7] (右腿接触) → 用于鼓励双腿着垫、避免单腿着陆
- action/engine: action 值本身 → 可惩罚非零动作（燃料使用）或针对性地惩罚姿态发动机频繁使用
- other: 无额外可用信号

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -121.00 | -121.00 | 0.00 | 68.35 | angle_penalty=-0.544 contact_quality=0.021 goal_proximity=-2.156 success_proxy=0.037 thrust_penalty=-0.024 | new_best |
| 2 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -183.32 | -121.00 | -62.32 | 947.35 | angle_penalty=-0.207 contact_quality=0.896 goal_proximity=-1.565 success_proxy=7.092 thrust_penalty=-0.040 | no_meaningful_improvement |
| 3 | angle_penalty + contact_quality + goal_proximity + success_proxy + thrust_penalty + velocity_penalty | -120.25 | -120.25 | 0.00 | 68.40 | angle_penalty=-0.563 contact_quality=0.022 goal_proximity=-2.285 success_proxy=0.037 thrust_penalty=-0.022 | new_best |
| 4 | angle_penalty + contact_quality + goal_proximity + soft_success + thrust_penalty + velocity_penalty | -1471.59 | -120.25 | -1351.33 | 435.40 | angle_penalty=-0.217 contact_quality=0.033 goal_proximity=0.946 soft_success=28.967 thrust_penalty=-0.044 | unsolved_stagnation_fresh_restart |
| 5 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -63.97 | -63.97 | 0.00 | 963.75 | goal_distance_progress=0.013 landing_contact_success=0.915 orientation_penalty=-0.014 soft_landing=-0.021 | new_best |
| 6 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -37.58 | -37.58 | 0.00 | 947.65 | goal_distance_progress=0.013 landing_contact_success=2.123 orientation_penalty=-0.016 soft_landing=-0.023 | new_best |
| 7 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | 156.75 | 156.75 | 0.00 | 577.95 | goal_distance_progress=0.014 landing_contact_success=1.327 orientation_penalty=-0.014 soft_landing=-0.022 | new_best |
| 8 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -18.99 | 156.75 | -175.74 | 1000.00 | goal_distance_progress=0.012 landing_contact_success=1.333 orientation_penalty=-0.016 soft_landing=-0.038 | no_meaningful_improvement |
| 9 | goal_distance_progress + landing_contact_success + orientation_penalty + soft_landing | -12.57 | 156.75 | -169.32 | 1000.00 | goal_distance_progress=0.015 landing_contact_success=1.505 orientation_penalty=-0.010 soft_landing=-0.004 | no_meaningful_improvement |
