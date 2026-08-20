# 1. Search objective
- target_score: 200.000000
- current_score: 170.641744
- gap_to_target: 29.358256
- target_achievement_ratio: 85.321%

# 2. 上一轮奖励函数代码（该轮得分: 170.641744）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 观测拆分
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft_contact = next_obs[6]
    nright_contact = next_obs[7]

    # 超参数 —— 仅 w_progress 由 5.0 → 20.0
    w_progress = 20.0        # 上调
    w_landing = 2.0
    w_land_vel = 10.0
    w_angle = 0.5
    w_angvel = 0.5
    engine_cost = 0.02

    # 距离计算
    dist = (x**2 + y**2) ** 0.5
    ndist = (nx**2 + ny**2) ** 0.5

    # 1. 主学习信号：距离改进（potential‑based shaping）
    progress = w_progress * (dist - ndist)

    # 2. 着陆质量软信号（连续因子 + 几何平均，包含接触偏置）
    altitude_factor = max(0.0, 1.0 - abs(ny) / 0.2)
    align_factor    = max(0.0, 1.0 - abs(nx) / 0.2)
    vx_factor       = max(0.0, 1.0 - abs(nvx) / 0.3)
    vy_factor       = max(0.0, 1.0 - abs(nvy) / 0.5)
    angle_factor    = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor  = 0.1 + 0.9 * (nleft_contact + nright_contact) * 0.5

    product = (altitude_factor * align_factor * vx_factor *
               vy_factor * angle_factor * contact_factor)
    if product > 0.0:
        landing_quality = w_landing * (product ** (1.0 / 6.0))
    else:
        landing_quality = 0.0

    # 3. 着陆速度惩罚（仅在接触时）
    fcontact = float(nleft_contact * nright_contact)
    if fcontact > 0.5:
        vel_pen = -w_land_vel * (nvx**2 + nvy**2)
    else:
        vel_pen = 0.0

    # 4. 姿态稳定惩罚（全程）
    att_penalty = -w_angle * (nangle**2) - w_angvel * (nangvel**2)

    # 5. 引擎使用惩罚（节省燃料）
    eng_pen = -engine_cost if action != 0 else 0.0

    total_reward = progress + landing_quality + vel_pen + att_penalty + eng_pen
    components = {
        "progress": progress,
        "landing_quality": landing_quality,
        "landing_velocity_penalty": vel_pen,
        "attitude_penalty": att_penalty,
        "engine_cost": eng_pen
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 817.15 | 99.05 | ✅ |
| 2 | 连续化的 landing_quality（含接触偏置）将在接近平台时提供稳定梯度，引导 agent 在最终阶段减速... | 连续化的 landing_quality（含接触偏置）将在接近平台时提供稳定梯度，引导 agent 在最终阶段减速... | 682.10 | 96.28 | ❌ |
| 3 | 将 progress 系数提升 4 倍，使其每步贡献与持续惩罚相当或更高，恢复 progress 作为主导引力，让... | 将 progress 系数提升 4 倍，使其每步贡献与持续惩罚相当或更高，恢复 progress 作为主导引力，让... | 363.30 | 170.64 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=170.641744, len=363.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-130.179203, 261.670162]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 38.508361 | 45.6% | 45.6% | 12.3% |
| progress | 21.076008 | 24.9% | 32.7% | 97.3% |
| landing_velocity_penalty | -6.959595 | -8.2% | 8.2% | 13.9% |
| attitude_penalty | -6.652874 | -7.9% | 7.9% | 100.0% |
| engine_cost | -4.804000 | -5.7% | 5.7% | 66.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score=170.64 over 20 episodes, all terminated (no truncations), mean ep_len=363.3. Generated reward per step is small (mean 0.54) but accumulates; original env reward is negative (-0.47/step). Total episode reward sums are moderate positive.

**Component Anomalies**: landing_quality dominates at 45.6% signed share but fires only 12.3% of steps — sparse, high-magnitude spikes. attitude_penalty fires 100% of steps but contributes only -7.9% (constant drag, not diagnostic). progress is the only reliable continuous signal: 97.3% active, 24.9% share.

**Training Dynamics**: No temporal dynamics available — component_dynamics returned no monitor snapshots. Cannot assess scaffold→final drift, early vs late activation profiles, or checkpoint-level trends.

**Signal Quality**: landing_quality sparsity (12.3% active) suggests a threshold or gate rarely crossed — potential reachability gap. landing_velocity_penalty similarly sparse (13.9%). attitude_penalty's 100% active rate makes it a constant offset, not a behavioral signal. No coupling analysis possible without dynamics.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个 2D 飞行器从初始位置（通常靠近视口顶部中央）出发，尽可能快地降落到场景中央的目标平台上，并以低速度、稳定姿态安全停稳，使两条支撑腿同时接触平台。次要目标：在完成任务的过程中，尽量减少引擎使用量（节省燃料、减少推力）。不应将姿态摆动最小化或单纯的速度最小化作为独立目标，这些只是达成安全着陆的附属约束。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32（匿名环境未明确，但符合连续观测惯例）
- 各维度含义：
  - obs[0]: x_position（水平坐标，相对于目标平台中心的偏移）—— reward_usable: true
  - obs[1]: y_position（垂直坐标，相对于平台高度基准的偏移）—— reward_usable: true
  - obs[2]: x_velocity（水平线速度）—— reward_usable: true
  - obs[3]: y_velocity（垂直速度）—— reward_usable: true
  - obs[4]: body_angle（机身倾斜角度）—— reward_usable: true
  - obs[5]: angular_velocity（角速度）—— reward_usable: true
  - obs[6]: left_support_contact（左侧支撑腿接触标志，1.0表示接触）—— reward_usable: true
  - obs[7]: right_support_contact（右侧支撑腿接触标志，1.0表示接触）—— reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：
  - action 0: no_engine —— 不启动任何引擎（惯性运动）
  - action 1: left_orientation_engine —— 启动左侧方向引擎（产生逆时针或顺时针力矩，改变姿态）
  - action 2: main_engine —— 启动主引擎（产生向上的推力，通常用于减速或上升）
  - action 3: right_orientation_engine —— 启动右侧方向引擎（产生与左侧引擎相反的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 为真，并且可以通过观测信号交叉验证：两条支撑腿均接触平台（obs[6] 和 obs[7] 都为 1.0）、水平位置接近 0（obs[0] ≈ 0）、垂直速度接近 0、姿态角接近水平。这种情况暗示飞行器已稳定停靠在目标平台上。
- **failure-like termination**：`crash_or_body_contact`（主体与地形或其他物体发生不期望的接触，导致损毁）、`horizontal_position_outside_viewport`（水平位置超出屏幕边界，飞行器脱离有效区域）。
- **ambiguous termination**：`body_not_awake_or_settled` 为真，但两条支撑腿未同时接触平台，或者位置不在平台附近。这可能是飞行器在平台外静止但未悬空（例如已经坠毁但引擎关闭或卡在地形中），需通过位置和接触信号判别。在初始学习阶段，部分此类终止可视为失败。
- **truncation**：无显式截断逻辑，`info` 为空，`truncated` 返回 False，即 episode 仅在触发上述终止条件时结束。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 内无任何成功标记）
- explicit_failure_flag_available: false（`info` 内无任何失败标记）
- allowed_info_fields: 无（`info` 返回空字典）
- forbidden_or_uncertain_info_fields: 所有 `info` 字段（因为没有声明任何可用字段，且原环境可能将奖励或终止原因隐藏在 `info` 中，但根据要求不能假设其存在，因此全部禁止使用）

## 7. 可用于奖励函数的信号
- **位置类**：`x_position` (obs[0])，`y_position` (obs[1])。可计算与目标平台的水平距离和垂直距离，用于引导接近。
- **速度类**：`x_velocity` (obs[2])，`y_velocity` (obs[3])。可用于惩罚着陆时的冲击速度，或在飞行阶段鼓励平滑性。
- **姿态类**：`body_angle` (obs[4])，`angular_velocity` (obs[5])。用于要求安全着陆时的姿态稳定性（尽量接近水平）。
- **接触类**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7])。两腿同时接触平台是成功着陆的必要条件，可据此构造着陆奖励。
- **动作/引擎类**：`action`。可惩罚引擎使用（no_engine 不惩罚，其余动作惩罚）以鼓励节省燃料。
- **派生推断信号（derived_possible）**：
  - 成功着陆指示器：可从 `body_not_awake_or_settled` 导致 episode 结束，且 `obs[6]` 和 `obs[7]` 均为 1.0、obs[0] 接近 0、obs[3] 接近 0 间接推断。可在奖励函数中结合 next_obs 构造着陆成功奖励，但需谨慎使用，因为无法直接读取终止原因。
  - 失败着陆指示器：可从 episode 结束时 `crash_or_body_contact` 或出界未接触双足推断，但同样无法在奖励计算时直接获取，只能通过观测模式判断（如 `next_obs` 中位置突变、速度极大等）。

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
| 1 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 99.05 | 99.05 | 0.00 | 817.15 | attitude_penalty=-0.042 engine_cost=-0.013 landing_quality=0.305 landing_velocity_penalty=-0.046 progress=0.033 | new_best |
| 2 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 96.28 | 99.05 | -2.76 | 682.10 | attitude_penalty=-0.059 engine_cost=-0.012 landing_quality=0.594 landing_velocity_penalty=-0.044 progress=0.032 | no_meaningful_improvement |
| 3 | attitude_penalty + engine_cost + landing_quality + landing_velocity_penalty + progress | 170.64 | 170.64 | 0.00 | 363.30 | attitude_penalty=-0.065 engine_cost=-0.014 landing_quality=0.525 landing_velocity_penalty=-0.047 progress=0.133 | new_best |
