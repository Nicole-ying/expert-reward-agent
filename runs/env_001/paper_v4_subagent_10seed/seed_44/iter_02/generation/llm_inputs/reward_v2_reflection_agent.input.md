# 1. Search objective
- target_score: 200.000000
- current_score: -119.080845
- gap_to_target: 319.080845
- target_achievement_ratio: -59.540%

# 2. 上一轮奖励函数代码（该轮得分: -119.080845）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations: both obs and next_obs are 8-dim vectors
    x,        y        = obs[0], obs[1]
    vx,       vy       = obs[2], obs[3]
    angle,    ang_vel  = obs[4], obs[5]
    # next_obs
    nx,       ny       = next_obs[0], next_obs[1]
    nvx,      nvy      = next_obs[2], next_obs[3]
    nangle,   nang_vel = next_obs[4], next_obs[5]
    lcon,     rcon     = next_obs[6], next_obs[7]  # contact flags at next state

    # ----- potential function (smaller values are better) -----
    def potential(px, py, pvx, pvy, pa):
        dist = (px**2 + py**2) ** 0.5
        vel  = (pvx**2 + pvy**2) ** 0.5
        return -(2.0 * dist + 1.0 * vel + 1.0 * abs(pa))

    # Main progress signal: improvement in potential
    pot_old = potential(x, y, vx, vy, angle)
    pot_new = potential(nx, ny, nvx, nvy, nangle)
    progress = pot_new - pot_old
    # Scale factor can be tuned, keep raw for now. Usually we want reward per step in range ~1.0
    main_progress = progress   # expected range roughly [-?..+?], but typical improvement gives ~0.1-1.0

    # ----- fuel efficiency (action cost) -----
    # action 0 = no engine, 1/2/3 = use engine
    fuel_penalty = -0.02 if action != 0 else 0.0

    # ----- extreme tilt hinge (hard safety) -----
    tilt = abs(nangle)
    tilt_limit = 0.5   # radians, strongly tilted
    if tilt > tilt_limit:
        extreme_tilt_penalty = -0.5 * (tilt - tilt_limit)
    else:
        extreme_tilt_penalty = 0.0

    # ----- soft contact encouragement (only when close to target) -----
    dist_to_target = (nx**2 + ny**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + dist_to_target)   # close → 1, far → 0
    contact_bonus = 0.2 * lcon * rcon * proximity_factor

    # ----- total reward -----
    total_reward = main_progress + fuel_penalty + extreme_tilt_penalty + contact_bonus

    components = {
        "potential_delta": main_progress,
        "fuel_penalty": fuel_penalty,
        "extreme_tilt_penalty": extreme_tilt_penalty,
        "stable_contact_bonus": contact_bonus
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-119.080845, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.524791, -98.176274]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| potential_delta | 2.060402 | 82.2% | 91.4% | 100.0% |
| stable_contact_bonus | 0.149898 | 6.0% | 6.0% | 1.3% |
| fuel_penalty | -0.066000 | -2.6% | 2.6% | 4.8% |
| extreme_tilt_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境为一个二维轨迹优化任务：代理人（飞行器）初始位于视口顶部中央附近，携带随机初始力。核心任务是**尽快到达并稳定停靠在画面中央的目标着陆垫上**，同时**尽可能减少发动机推力使用**。代理人需要学会接近目标、减速、保持姿态稳定并实现安全接触。

主要目标：在尽量短的时间内，让身体相对着陆垫的位置（x, y）趋近于零，同时速度降为零，姿态保持竖直（body_angle≈0），并使左右两个支撑脚同时接触。  
次要目标：最小化燃料消耗（动作中使用发动机的次数）。  
不应混淆为仅需靠近即可得分的悬浮任务——单纯悬浮不应获得持续奖励，且最终必须实现有接触的稳定停靠。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64（推断为 float，实际代码中可能 float32/float64，对奖励无影响）  
- 各维度含义（均基于 next_obs 视角，无历史滑动窗口）：

| 索引 | 名称                     | 含义                                                                 | reward_usable |
|------|--------------------------|----------------------------------------------------------------------|---------------|
| 0    | x_position               | 身体水平坐标，相对于着陆垫中心的偏移                                   | true          |
| 1    | y_position               | 身体垂直坐标，相对于着陆垫高度的偏移                                   | true          |
| 2    | x_velocity               | 水平线速度                                                            | true          |
| 3    | y_velocity               | 垂直线速度                                                            | true          |
| 4    | body_angle               | 身体朝向角（弧度）                                                    | true          |
| 5    | angular_velocity         | 角速度                                                                | true          |
| 6    | left_support_contact     | 左支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |
| 7    | right_support_contact    | 右支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作表：

| 动作 id | 名称                        | 含义                                                         |
|--------|-----------------------------|------------------------------------------------------------|
| 0      | no_engine                   | 不启动任何引擎，自由漂移                                    |
| 1      | left_orientation_engine     | 点燃左侧姿态引擎（产生逆时针？力矩，用于调整姿态）           |
| 2      | main_engine                 | 点燃主引擎（产生垂直向上的推力？或向下的推力？根据相对坐标系，可能提供垂直方向推力抵消重力/加速） |
| 3      | right_orientation_engine    | 点燃右侧姿态引擎（产生与左侧相反的力矩）                     |

注：虽然动作空间为离散，但动力学为连续（位置、速度、角度）。主引擎和姿态引擎的具体推力方向由底层物理决定，奖励函数只需知道动作 ID 即可识别是否使用了推力（id ≠ 0 时为有燃料消耗的动作）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  - `body_not_awake_or_settled` 中的 *settled* 状态：当身体处于静止、双支撑脚接触、且位置/姿态满足一定标准时，被判为已停稳，终止回合。此即任务成功的信号。  
  - 从观测推断：若终止发生时，左右接触标志均为 1，位置 (0,0) 附近，角度≈0，速度≈0，则极大概率为成功。
- **failure-like termination**:  
  - `crash_or_body_contact`：身体与不可碰撞部位（如地面或非着陆垫物体）发生接触，或除支撑脚外的部位触地。  
  - `horizontal_position_outside_viewport`：水平位置超出视野边界。  
  - `body_not_awake_or_settled` 中的 *body_not_awake*：身体失去“意识”（可能因高速撞击、翻滚导致），但并非稳定停泊，属于失败。
- **ambiguous termination**:  
  - 仅有终止信号，没有显式 success/failure 标志时，需根据最终观测状态判断成败。  
  - `body_not_awake_or_settled` 内部可能包含成功（settled）和失败（not awake），完全依赖于观测解读。
- **truncation**: 本描述中未见最大步长截断（MASKED_STEP_SOURCE 中 `truncated=False` 始终返回），因此所有终止均为 terminated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 根据 step 源码，info 字典为空，**无任何可用字段**。  
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用；不得假设存在 `success`、`failure`、`termination_reason` 等。

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
| 1 | extreme_tilt_penalty + fuel_penalty + potential_delta + stable_contact_bonus | -119.08 | -119.08 | 0.00 | 68.30 | extreme_tilt_penalty=-0.001 fuel_penalty=-0.003 potential_delta=0.029 stable_contact_bonus=0.003 | new_best |
