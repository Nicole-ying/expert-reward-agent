# 1. Search objective
- target_score: 200.000000
- current_score: 97.698965
- gap_to_target: 102.301035
- target_achievement_ratio: 48.849%

# 2. 上一轮奖励函数代码（该轮得分: 97.698965）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation variables
    x, y = obs[0], obs[1]
    x_v, y_v = obs[2], obs[3]
    angle = obs[4]
    ang_v = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nx_v, ny_v = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_v = next_obs[5]
    n_left = next_obs[6]
    n_right = next_obs[7]

    # ---------- 1. Progress reward: moving toward the landing pad (0,0) ----------
    dist_curr = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_curr - dist_next          # positive when getting closer
    progress_reward = 1.0 * progress          # weight = 1.0

    # ---------- 2. Horizontal boundary penalty (crash prevention) ----------
    x_limit = 1.2
    x_boundary_penalty = 0.5 * max(0.0, abs(nx) - x_limit)

    # ---------- 3. Landing softness / safety penalty ----------
    # Velocity and angular velocity limits
    v_limit = 0.5
    vx_pen = max(0.0, abs(nx_v) - v_limit)
    vy_pen = max(0.0, abs(ny_v) - v_limit)
    vel_pen = vx_pen + vy_pen

    ang_limit = 1.0
    ang_pen = max(0.0, abs(n_ang_v) - ang_limit)

    tilt_pen = abs(n_angle)                  # ideal angle is 0

    # Distance‑based activation gate: only enforce strict softness near the pad
    gate = 1.0 / (1.0 + 5.0 * dist_next)     # increases when close to target

    landing_safety_penalty = (0.1 * vel_pen + 0.05 * ang_pen + 0.1 * tilt_pen) * gate

    # ---------- 4. Landing contact bonus: positive signal for proper touchdown ----------
    # Reward both legs contacting the ground, gated to only activate near the target
    landing_contact_bonus = 0.3 * (n_left + n_right) * gate

    # ---------- Total reward ----------
    total_reward = progress_reward - x_boundary_penalty - landing_safety_penalty + landing_contact_bonus

    components = {
        "progress_reward": float(progress_reward),
        "x_boundary_penalty": float(x_boundary_penalty),
        "landing_safety_penalty": float(landing_safety_penalty),
        "landing_contact_bonus": float(landing_contact_bonus)
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 211.25 | 39.61 | ✅ |
| 2 | 新增接触奖励让agent有动力在接近目标时保持双腿着地姿态，配合landing_safety_penalty抑制速... | 新增接触奖励让agent有动力在接近目标时保持双腿着地姿态，配合landing_safety_penalty抑制速... | 872.65 | 97.70 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=97.698965, len=872.650000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-325.840631, 185.591390]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact_bonus | 365.465579 | 99.2% | 99.2% | 72.3% |
| progress_reward | 1.295489 | 0.4% | 0.4% | 100.0% |
| landing_safety_penalty | 1.399652 | 0.4% | 0.4% | 100.0% |
| x_boundary_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: 1) Total reward (0.283) is almost entirely landing_contact_bonus (0.2834, 64.2% active), while progress_reward is negligible (0.0031, 100% active). 2) Only 3/20 episodes terminate (15%) despite high total reward; original_env_reward is negative (-0.0209). 3) x_boundary_penalty is dead (0% activation, mean 0.0) — the agent never exceeds |x|>1.2.

**Component Anomalies**: x_boundary_penalty: completely dead (0% nonzero_rate, 0.0 mean) — never triggers and contributes nothing to learning. landing_contact_bonus: dominates at ~100x the magnitude of progress_reward, likely drowning out the distance-minimization signal.

**Mechanism Hypothesis**: The landing_contact_bonus (0.3 × Σcontact × gate) overwhelms the progress_reward (1.0 × Δdistance), causing the agent to pursue ground contact wherever possible rather than steering toward the pad at (0,0). This explains high reward with low termination: the agent gets contact bonuses without actually reaching the target and completing the episode.

**Decision Implication**: PATCH landing_contact_bonus: reduce its coefficient from 0.3 to ~0.05 or make the gate sharper (e.g., gate² or require dist<0.3) so contact reward only matters very close to the pad. REMOVE x_boundary_penalty: it never fires, so it's dead weight in the reward function.

**Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境要求一个 2D 飞行器（带主引擎和两个姿态引擎）从视口顶部中心附近出发，以随机初始速度开始，尽快到达视口中心的着陆平台，并以极低的速度、稳定的姿态安全接触并停稳。主目标是**精确到达目标位置并稳定停靠**；次要目标是**快速完成**和**尽可能少地使用引擎推力**。任务的核心是导航与精确着陆，不应与纯粹的生存或持续前进任务混淆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- 各维含义：
  - obs[0]（x_position）：水平坐标，相对于目标着陆点的水平偏移，reward_usable: true
  - obs[1]（y_position）：垂直坐标，相对于平台高度的垂直偏移（平台高度处为 0），reward_usable: true
  - obs[2]（x_velocity）：水平线速度，reward_usable: true
  - obs[3]（y_velocity）：垂直线速度，reward_usable: true
  - obs[4]（body_angle）：机体朝向角（弧度），reward_usable: true
  - obs[5]（angular_velocity）：角速度，reward_usable: true
  - obs[6]（left_support_contact）：左侧支撑杆与地面/平台的接触标志（1.0 表示接触），reward_usable: true
  - obs[7]（right_support_contact）：右侧支撑杆接触标志（1.0 表示接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作列表：
  - action 0: no_engine，不开启任何引擎
  - action 1: left_orientation_engine，开启左姿态引擎（产生角加速度，可能向左旋转）
  - action 2: main_engine，开启主引擎（产生向上的推力）
  - action 3: right_orientation_engine，开启右姿态引擎（产生相反方向的角加速度）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  机体在平台上稳定停靠，触发 **body_not_awake_or_settled**（休眠/静止）。此时通常同时满足：两腿接触 flag 均为 1，x_position 和 y_position 接近 0，线速度与角速度均很小，且未发生 crash 或出界。
- failure-like termination:  
  - **horizontal_position_outside_viewport**：机体水平飞出视口边界，直接失败。  
  - **crash_or_body_contact**（非着陆接触）：机体以过大速度、过大角度或接触到非平台区域（如地面以外）触发终止，属于失败。需要结合接触标志和速度判断。
- ambiguous termination:  
  **crash_or_body_contact** 在某些情况下也可能是成功着陆，因为着陆时也会发生身体接触并可能触发该条件。需要进一步通过双腿是否都接触、速度是否低、是否在目标附近来区分。  
  **body_not_awake_or_settled** 也可能是碰撞后卡住不动导致的静止，但碰撞后通常接触标志不会全为 1 且位置会偏离目标，因此可通过位置与接触标志排除模糊性。
- truncation:  
  无显式截断（源码中返回的 truncated 恒为 False）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 始终为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用
- 推断成功/失败的间接路径（derived_possible）：
  - **成功**：episode 终止且满足以下条件 → (left_support_contact == 1.0) and (right_support_contact == 1.0) and (|x_position| 很小) and (|y_position| 很小) and (|body_angle| 很小) and 线速度/角速度很低。  
  - **失败（crash）**：episode 终止但上述条件不成立（例如双腿未同时接触、位置大幅偏离、角度或速度很大）。  
  - **出界**：可通过终止时 |x_position| 显著大于视口半宽推断。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（均可直接获得，表示相对于目标的位置）
- velocity: x_velocity, y_velocity, angular_velocity
- orientation: body_angle
- contact: left_support_contact, right_support_contact
- action/engine: 离散动作 id 可映射为推力状态（无推力、左旋、主推、右旋）；可用于估计燃料消耗、避免无用点火
- other: 从上述信号可派生的距离（euclidean distance, |x|+|y| 等）、接近速度、朝向对齐程度、双腿是否均接触、是否在目标附近等

所有信号均为可直接从 obs 或 next_obs 读取的数值，无量纲但具有物理意义。

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_safety_penalty + progress_reward + x_boundary_penalty | 39.61 | 39.61 | 0.00 | 211.25 | landing_safety_penalty=0.003 progress_reward=0.007 x_boundary_penalty=0.000 | new_best |
| 2 | landing_contact_bonus + landing_safety_penalty + progress_reward + x_boundary_penalty | 97.70 | 97.70 | 0.00 | 872.65 | landing_contact_bonus=0.283 landing_safety_penalty=0.003 progress_reward=0.003 x_boundary_penalty=0.000 | new_best |
