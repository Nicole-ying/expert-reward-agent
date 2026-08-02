# 1. Search objective
- target_score: 200.000000
- current_score: 125.329636
- gap_to_target: 74.670364
- target_achievement_ratio: 62.665%

# 2. 上一轮奖励函数代码（该轮得分: 125.329636）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices (same as before)
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle, n_ang_vel = next_obs[4], next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # ---------- 1. Main progress: distance to target ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. Stability ----------
    w_angle = 0.5
    w_angvel = 0.1
    stability = -w_angle * (n_angle ** 2) - w_angvel * (n_ang_vel ** 2)

    # ---------- 3. Lateral drift ----------
    w_lat = 0.1
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Landing approach (continuous, replaces sparse landing_bonus) ----------
    # Ground proximity: y near 0 (assumed pad altitude) gives high factor
    ground_prox = 1.0 / (1.0 + ny ** 2)
    # Attitude factor: angle within 0.3 rad
    angle_factor = max(0.0, 1.0 - abs(n_angle) / 0.3)
    # Vertical speed factor: gentle vertical speed
    vy_factor = max(0.0, 1.0 - abs(nvy) / 0.3)
    landing_factor = (ground_prox + angle_factor + vy_factor) / 3.0
    w_landing = 0.05
    landing_approach = w_landing * landing_factor

    # ---------- 5. Descending penalty ----------
    safe_vy = -0.3
    w_desc = 0.5
    descending_penalty = w_desc * min(0.0, nvy - safe_vy)

    # Combine
    total_reward = (w_progress * progress
                    + stability
                    + lateral_drift
                    + landing_approach
                    + descending_penalty)

    components = {
        "progress": w_progress * progress,
        "stability_penalty": stability,
        "lateral_drift_penalty": lateral_drift,
        "landing_approach": landing_approach,
        "descending_penalty": descending_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 101.60 | -111.37 | ✅ |
| 2 | 加入下降速度 hinge 惩罚后，agent 将被迫学习控制下降速度，延长存活时间，增加发现 landing_bo... | 加入下降速度 hinge 惩罚后，agent 将被迫学习控制下降速度，延长存活时间，增加发现 landing_bo... | 1000.00 | -62.36 | ✅ |
| 3 | 连续着陆因子会在降落过程中提供梯度信号，引导 agent 降低高度并维持姿态，最终触发真实着陆。 | 连续着陆因子会在降落过程中提供梯度信号，引导 agent 降低高度并维持姿态，最终触发真实着陆。 | 1000.00 | 125.33 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=125.329636, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[79.069905, 161.303778]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 43.781036 | 71.3% | 71.3% | 100.0% |
| progress | 13.735301 | 22.4% | 23.7% | 100.0% |
| lateral_drift_penalty | -1.395423 | -2.3% | 2.3% | 100.0% |
| stability_penalty | -1.269045 | -2.1% | 2.1% | 100.0% |
| descending_penalty | -0.415939 | -0.7% | 0.7% | 1.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


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
| 2 | descending_penalty + landing_bonus + lateral_drift_penalty + progress + stability_penalty | -62.36 | -62.36 | 0.00 | 1000.00 | descending_penalty=-0.066 landing_bonus=3.117 lateral_drift_penalty=-0.007 progress=0.045 stability_penalty=-0.017 | new_best |
| 3 | descending_penalty + landing_approach + lateral_drift_penalty + progress + stability_penalty | 125.33 | 125.33 | 0.00 | 1000.00 | descending_penalty=-0.024 landing_approach=0.040 lateral_drift_penalty=-0.003 progress=0.024 stability_penalty=-0.009 | new_best |
