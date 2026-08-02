# 1. Search objective
- target_score: 200.000000
- current_score: -24.964255
- gap_to_target: 224.964255
- target_achievement_ratio: -12.482%

# 2. 上一轮奖励函数代码（该轮得分: -24.964255）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, ang_vel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, n_angle, n_ang_vel, n_left, n_right = next_obs

    # ---------- 1. Main progress: improvement in Euclidean distance to landing pad ----------
    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    delta_dist = dist - next_dist                # positive when getting closer
    progress_reward = 2.0 * delta_dist

    # ---------- 2. Attitude safety constraint ----------
    angle_err = abs(n_angle)
    ang_vel_abs = abs(n_ang_vel)
    attitude_penalty = -0.5 * (angle_err**2 + (0.5 * ang_vel_abs)**2)

    # ---------- 3. Landing approach reward (continuous multi-factor, replaces dead success_reward) ----------
    prox = max(0.0, 1.0 - next_dist / 5.0)
    upright = max(0.0, 1.0 - angle_err / 0.5)
    speed = (nvx**2 + nvy**2) ** 0.5
    stationary = max(0.0, 1.0 - speed / 1.0)
    contact = (n_left + n_right) / 2.0
    landing_reward = 1.0 * (prox + upright + stationary + contact) / 4.0

    # ---------- Aggregate ----------
    total_reward = progress_reward + attitude_penalty + landing_reward

    components = {
        "progress_reward": progress_reward,
        "attitude_penalty": attitude_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 1000.00 | -36.03 | ✅ |
| 2 | 骨架变化: attitude_penalty + landing_reward + progress_rewar | — | 1000.00 | -24.96 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-24.964255, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-50.610096, 13.490928]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 686.511964 | 99.5% | 99.5% | 100.0% |
| progress_reward | 2.267035 | 0.3% | 0.4% | 100.0% |
| attitude_penalty | -1.082570 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Landing_reward dominates (99.5% signed share, ep_sum_mean=686.5) yet final score=-24.96, 0/20 terminated (all truncated at 1000 steps). Progress and attitude components negligible.

**Component Anomalies**: Landing_reward >99% share, not dead. Attitude_penalty mean=-0.0097, near-zero share. No component >70% magnitude share (landing_reward magnitude share 99.5% = dominating).

**Training Dynamics**: No temporal monitor snapshots provided; drift across checkpoints unknown.

**Signal Quality**: All components active 100%, no dead gates. Landing_reward signal fails to induce terminal landings; episodes never terminate early despite high reward sums.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D飞行器精确着陆任务。agent 从视野顶部中心附近出发，带有随机初始扰动。核心目标为安全、稳定地在中心目标平台上着陆——即到达指定相对水平位置 x≈0、高度 y≈0（平台高度），同时保持姿态接近竖直、双腿同时接触平台、速度几乎为零。次要目标为尽快完成着陆，并尽量少使用引擎推力（降低燃料消耗）。不应将存活时间或长时间悬停作为正面目标，也不应单纯最大化水平进度而忽略触地质量和姿态约束。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（或 float64）
- obs[0]: x_position，相对于目标平台水平坐标，reward_usable: true
- obs[1]: y_position，相对于平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，身体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不激活任何引擎（保持当前惯性）
- action 1: left_orientation_engine，点燃左朝向引擎（产生转向或侧向推力）
- action 2: main_engine，点燃主引擎（一般提供向上的推力，但也可能产生旋转分量）
- action 3: right_orientation_engine，点燃右朝向引擎（转向或侧向推力，方向与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- crash_or_body_contact：身体（除双腿外的部分）与地面发生碰撞 → 很可能为失败终止（坠毁）。
- horizontal_position_outside_viewport：水平位置超出视野边界 → 失败终止（出界）。
- body_not_awake_or_settled：身体不再活跃（例如静止且未触发其他终止）或满足平台稳定着陆条件（settled） → 若为 settled 则属于成功终止，若仅为不活跃但未满足着陆要求则可能为中立或失败。从任务目标推断，成功着陆的唯一途径就是触发 settled 条件（双腿接触、速度极低、姿态竖直等），因此该条件可视为成功类终止，但需要谨慎对待可能的非成功不活跃情形。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无 success 字段，原始观测亦无直接标志）
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典，禁止读取任何字段）
- forbidden_or_uncertain_info_fields: info 内的任何内容均不可用；禁止使用 original_reward

补充推断路径（derived_possible）：
- 成功着陆可通过“终止时的 next_obs 满足两条腿接触、速度接近零、角度接近零、且未发生 crash 或出界”间接判断。
- 坠毁可通过突然的高加速度、body_angle 突变、或 body 位置骤然下降并伴随 contact 信号异常间接推断。
- 出界可从 x_position 超出视野范围推测。

## 7. 可用于奖励函数的信号
位置相关：
- x_position, y_position（均可用于计算到目标点的欧氏距离、水平偏移、高度偏差；可构造距离进步量 delta_distance）
- 可通过 next_obs 与 obs 的 x/y 位置差获取位移方向

速度相关：
- x_velocity, y_velocity（可用于惩罚水平漂移、过大的垂直速度，特别是在接近目标时；可构造速度门控惩罚）
- 速度平方/模长可用于能量惩罚

姿态相关：
- body_angle（用于惩罚偏离竖直的姿态，着陆阶段应接近 0）
- angular_velocity（惩罚过大角速度，防止剧烈旋转）

接触相关：
- left_support_contact, right_support_contact（用于鼓励双腿同时接地，或惩罚单脚/belly着陆）

动作相关：
- action 的语义（no_engine、主引擎、偏转引擎）可用于燃料惩罚（如非零动作施加小惩罚）

间接推断成功的信号（derived_possible）：
- 当 next_obs 满足：双腿接触均为 1、x_velocity≈0、y_velocity≈0、|body_angle| ≈0、x_position≈0、y_position≈0，且当前步未检测到 crash 条件时，可以高置信度推断着陆成功，用于终端奖励。

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
| 1 | attitude_penalty + progress_reward + success_reward | -36.03 | -36.03 | 0.00 | 1000.00 | attitude_penalty=-0.022 progress_reward=0.010 success_reward=1.686 | new_best |
| 2 | attitude_penalty + landing_reward + progress_reward | -24.96 | -24.96 | 0.00 | 1000.00 | attitude_penalty=-0.010 landing_reward=0.643 progress_reward=0.002 | new_best |
