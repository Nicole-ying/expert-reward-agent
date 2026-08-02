# 1. Search objective
- target_score: 200.000000
- current_score: -103.621729
- gap_to_target: 303.621729
- target_achievement_ratio: -51.811%

# 2. 上一轮奖励函数代码（该轮得分: -103.621729）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Observation indices
    # 0: x position, 1: y position, 2: vx, 3: vy, 4: angle, 5: angular velocity
    # 6: left leg contact, 7: right leg contact (0.0 or 1.0)

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
    progress = dist_old - dist_new

    w_progress = 10.0

    # ---------- 2. Attitude gate: suppress progress when angle is dangerous ----------
    # Smooth hinge: gate ≈ 1.0 when |angle| << 0.15, gate → 0.2 when |angle| >> 0.15
    # Use tanh for smooth transition; 0.15 rad ≈ 8.6° is safety threshold
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. Lateral drift constraint: horizontal speed ----------
    w_lat = 0.2
    lateral_drift = -w_lat * (nvx ** 2)

    # ---------- 4. Angular velocity penalty: small auxiliary smoothing ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. Landing bonus: soft continuous proxy ----------
    # Both legs touching, nearly upright, gentle speeds
    both_legs = min(n_left, n_right)              # 0.0 to 1.0
    vertical_ok = max(0.0, 1.0 - abs(nvy) / 0.3) # 1.0 when vy≈0, 0 when |vy|>=0.3
    attitude_ok = max(0.0, 1.0 - abs_angle / 0.15) # 1.0 when angle≈0, 0 when |angle|>=0.15

    landing_factor = both_legs * vertical_ok * attitude_ok
    landing_bonus = 3.0 * landing_factor          # up to 3.0, smooth

    # Combine: progress is gated by attitude, then penalties and bonus added
    total_reward = (w_progress * progress * angle_gate
                    + lateral_drift
                    + angvel_penalty
                    + landing_bonus)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_drift_penalty": lateral_drift,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 101.60 | -111.37 | ✅ |
| 2 | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 68.70 | -103.62 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-103.621729, len=68.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.881582, -80.458117]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 10.572382 | 65.2% | 67.6% | 100.0% |
| lateral_drift_penalty | -2.917951 | -18.0% | 18.0% | 99.7% |
| landing_bonus | 1.257927 | 7.8% | 7.8% | 0.7% |
| angvel_penalty | -1.077620 | -6.6% | 6.6% | 99.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| 2 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -103.62 | -103.62 | 0.00 | 68.70 | angvel_penalty=-0.020 landing_bonus=0.016 lateral_drift_penalty=-0.039 progress_gated=0.144 | new_best |
