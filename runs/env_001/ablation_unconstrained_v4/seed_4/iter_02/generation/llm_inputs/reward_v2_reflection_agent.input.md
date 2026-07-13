# Duplicate reward retry
The previous generation duplicated iter 1 (runs\env_001\ablation_unconstrained_v4\seed_4\iter_01\generation\reward_v1.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 1.0   # distance reduction weight
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_bonus  = 0.5   # stable‑stand bonus weight

    k_gate_y = 20.0  # steepness of height gate
    k_fx     = 5.0   # factor scaling for horizontal distance
    k_fy     = 5.0   # factor scaling for vertical height
    k_fspeed = 2.0   # factor scaling for speed
    k_fangle = 2.0   # factor scaling for body angle

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    # gate = 1 when y is near zero (height close to pad), decays when high
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. stable‑stand bonus (joint condition proxy) ----------
    # each factor ∈ [0,1]; geometric mean avoids collapse
    factor_x       = 1.0 / (1.0 + k_fx * abs(next_obs[0]))
    factor_y       = 1.0 / (1.0 + k_fy * max(0.0, next_obs[1]))
    factor_speed   = 1.0 / (1.0 + k_fspeed * speed)
    factor_angle   = 1.0 / (1.0 + k_fangle * abs(next_obs[4]))
    factor_contact = next_obs[6] * next_obs[7]  # 1 only when both legs touch

    prod = factor_x * factor_y * factor_speed * factor_angle * factor_contact
    stable_bonus = w_bonus * (prod ** 0.2) if prod > 0.0 else 0.0

    # ---------- total reward ----------
    total_reward = progress_reward + landing_vel_penalty + upright_penalty + stable_bonus

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'stable_bonus': stable_bonus
    }
    return float(total_reward), components
```

# Search objective
- target_score: 200.000000
- current_score: -101.892341
- gap_to_target: 301.892341
- target_achievement_ratio: -50.946%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -101.892341）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox   = 1.0   # distance reduction weight
    w_vel    = 0.3   # landing velocity penalty weight
    w_angle  = 0.2   # body angle penalty weight
    w_angvel = 0.1   # angular velocity penalty weight
    w_bonus  = 0.5   # stable‑stand bonus weight

    k_gate_y = 20.0  # steepness of height gate
    k_fx     = 5.0   # factor scaling for horizontal distance
    k_fy     = 5.0   # factor scaling for vertical height
    k_fspeed = 2.0   # factor scaling for speed
    k_fangle = 2.0   # factor scaling for body angle

    # ---------- 1. progress: distance reduction ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress  = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. soft landing velocity penalty ----------
    # gate = 1 when y is near zero (height close to pad), decays when high
    y_clipped = max(0.0, next_obs[1])   # y positive above pad
    gate_vel  = 1.0 / (1.0 + k_gate_y * y_clipped)
    speed     = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    landing_vel_penalty = -w_vel * speed * gate_vel

    # ---------- 3. upright stability penalty ----------
    upright_penalty = -w_angle * abs(next_obs[4]) - w_angvel * abs(next_obs[5])

    # ---------- 4. stable‑stand bonus (joint condition proxy) ----------
    # each factor ∈ [0,1]; geometric mean avoids collapse
    factor_x       = 1.0 / (1.0 + k_fx * abs(next_obs[0]))
    factor_y       = 1.0 / (1.0 + k_fy * max(0.0, next_obs[1]))
    factor_speed   = 1.0 / (1.0 + k_fspeed * speed)
    factor_angle   = 1.0 / (1.0 + k_fangle * abs(next_obs[4]))
    factor_contact = next_obs[6] * next_obs[7]  # 1 only when both legs touch

    prod = factor_x * factor_y * factor_speed * factor_angle * factor_contact
    stable_bonus = w_bonus * (prod ** 0.2) if prod > 0.0 else 0.0

    # ---------- total reward ----------
    total_reward = progress_reward + landing_vel_penalty + upright_penalty + stable_bonus

    components = {
        'progress': progress_reward,
        'landing_vel_penalty': landing_vel_penalty,
        'upright_penalty': upright_penalty,
        'stable_bonus': stable_bonus
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-101.892341, len=68.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-121.144123, -73.274144]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_vel_penalty | -2.747915 | -56.7% | 56.7% | 100.0% |
| progress | 1.119153 | 23.1% | 23.9% | 100.0% |
| stable_bonus | 0.471253 | 9.7% | 9.7% | 1.9% |
| upright_penalty | -0.468024 | -9.7% | 9.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个2D平面内的着陆器轨迹优化任务。智能体初始位于视口顶部中央区域，带有随机初始力。核心目标是**到达并稳定地停泊在画面中央的目标垫板上**，同时尽可能减少引擎推力总消耗，并且在接触垫板时保持低速度和接近直立的姿态。次要目标包括：减少燃料使用、缩短完成任务时间，以及避免任何不安全接触。该任务**不是**持续的行走推进，也不是无明确到达目标的平衡存活，也不是多目标权重均等的复合任务；一切奖励设计最终服务于 “到达、减速、稳定着陆” 这一中心目的。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（各维度均为浮点，接触标志以 0.0 或 1.0 的浮点形式出现）  
- 各维度含义与奖励可用性：  
  - obs[0]: x_position（相对目标垫板的水平坐标），reward_usable: true  
  - obs[1]: y_position（相对垫板高度的垂直坐标），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体姿态角），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支撑点接触标志，1.0=已接触），reward_usable: true  
  - obs[7]: right_support_contact（右支撑点接触标志，1.0=已接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义（均为离散动作，不涉及连续幅度）：  
  - action 0: no_engine（不点火任何引擎）  
  - action 1: left_orientation_engine（点燃左侧定向引擎，主要用于顺时针或逆时针方向调整）  
  - action 2: main_engine（点燃主引擎，产生沿机体轴向的推力）  
  - action 3: right_orientation_engine（点燃右侧定向引擎，作用与左侧引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  身体稳定停泊在目标垫板上（body_not_awake_or_settled 触发），且未发生 crash 或出界。  
- failure-like termination:  
  crash_or_body_contact（与地面或非目标区域的硬接触）、horizontal_position_outside_viewport（水平位置超出视口边界）。  
- ambiguous termination:  
  无特别模糊的情况，但**单独**依靠 body_not_awake_or_settled 无法区分成功与因卡死在错误位置而“静止”，需要结合位置和接触信号共同判断；另外在训练早期，可能因初始随机力漂移导致未能接触垫板而直接出界。  
- truncation:  
  未在描述中明确提及最大步数截断，但环境可能在超时后终止，具体截断逻辑被遮蔽，不应在奖励函数中隐式利用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: info 为固定空字典 `{}`，**没有任何可用字段**。  
- forbidden_or_uncertain_info_fields: 官方 success/failure 标志、终止原因字符串等均不可用，因为步函数返回 info 为空。

## 7. 可用于奖励函数的信号
- position: 可通过 obs[0], obs[1] 及 next_obs[0], next_obs[1] 计算相对于目标的距离或位置变化。  
- velocity: 可通过 obs[2], obs[3] 与 next_obs[2], next_obs[3] 得到合速度大小或各分量。  
- orientation: obs[4], next_obs[4] 提供机体角度；obs[5], next_obs[5] 提供角速度。  
- contact: obs[6], obs[7] 与 next_obs[6], next_obs[7] 为两腿接触标志，可用于判定着陆状态。  
- action/engine: 动作序号可用于判断是否点燃主引擎（action==2）或定向引擎（action==1 或 3），从而计算燃料使用。  
- other: 由位置、速度、角度衍生出距离变化率（接近速度）、姿态变化等合成信号。

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_vel_penalty + progress + stable_bonus + upright_penalty | -101.89 | -101.89 | 0.00 | 68.70 | landing_vel_penalty=-0.043 progress=0.016 stable_bonus=0.006 upright_penalty=-0.011 | new_best |
