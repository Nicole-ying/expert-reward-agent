# 1. Search objective
- target_score: 200.000000
- current_score: -111.374634
- gap_to_target: 311.374634
- target_achievement_ratio: -55.687%

# 2. 上一轮奖励函数代码（该轮得分: -111.374634）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (assumed 2D lander, similar to LunarLander‑v2)
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)
    # Target landing pad is assumed at (0.0, 0.0)

    # Extract current and next states
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target decreasing ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new              # + when approaching target

    w_progress = 10.0

    # ---------- 2. Stability constraint: attitude & angular velocity ----------
    # Quadratic penalty on tilt and rotation
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Soft landing bonus (joint‑condition proxy) ----------
    landing_bonus = 0.0
    # Conditions: both legs on ground, nearly upright, gentle vertical speed
    if n_left > 0.5 and n_right > 0.5 and abs(n_angle) < 0.1 and abs(nvy) < 0.2:
        landing_bonus = 10.0

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_bonus)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-111.374634, len=101.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-153.325469, -83.366365]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 12.446921 | 50.5% | 65.6% | 100.0% |
| landing_bonus | 3.500000 | 14.2% | 14.2% | 0.3% |
| stability_penalty | -2.810825 | -11.4% | 11.4% | 100.0% |
| lateral_drift_penalty | -2.173843 | -8.8% | 8.8% | 99.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Score=-111.37, all episodes terminated but 80% early (<150 steps, score<-50). Shaped reward per-step positive (0.14) but original env reward negative (-1.24). Progress dominates (50.5% signed share), landing bonus almost never active (0.3% active rate).

**Component Anomalies**: landing_bonus dead: active rate 0.3%, episode sum mean 3.5 from sporadic triggers. Progress has 100% nonzero, dominates magnitude share (65.6%). Stability and lateral drift penalties always active but low magnitude shares (11.4%, 8.8%).

**Training Dynamics**: No monitor snapshots; no temporal trend data available across checkpoints.

**Signal Quality**: Dead gate: landing_bonus rarely meets thresholds (both legs contact, |angle|<0.1, |vy|<0.2). Progress always positive, failing to distinguish successful landing from uncontrolled descent. Missing attractor for soft landing behavior. No coupling analysis possible from this data.

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近启动，受到随机初始外力。主要目标是**尽快、平稳地降落在中央的目标平台上**（即到达并稳定停留在着陆点），同时

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
| 1 | landing_bonus + lateral_drift_penalty + progress + stability_penalty | -111.37 | -111.37 | 0.00 | 101.60 | landing_bonus=0.057 lateral_drift_penalty=-0.033 progress=0.147 stability_penalty=-0.028 | new_best |
