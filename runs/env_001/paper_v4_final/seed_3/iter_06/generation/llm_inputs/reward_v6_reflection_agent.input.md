# 1. Search objective
- target_score: 200.000000
- current_score: 30.056993
- gap_to_target: 169.943007
- target_achievement_ratio: 15.028%

# 2. 上一轮奖励函数代码（该轮得分: 30.056993）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_lc = next_obs[6]  # left contact
    n_rc = next_obs[7]  # right contact

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    # Relaxed from 0.15 to 0.30 (17°) to allow reasonable maneuvering
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.15
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy ----------
    # Only rewards when legs actually touch the ground, with safety gates
    mean_contact = (n_lc + n_rc) / 2.0  # [0, 1] continuous
    
    # Safety gates: y close to 0, low vy, low angle
    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))
    
    # Geometric mean of contact and safety factors
    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.25
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 101.60 | -111.37 | ✅ |
| 2 | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 将二次稳定性惩罚改为角度门控（gate进步奖励），在危险角度区域强力抑制progress，迫使agent学习保持小... | 68.70 | -103.62 | ✅ |
| 3 | 将接触条件从 `min` 改为均值，使单腿触地也能获得部分奖励，提供连续梯度；agent 将开始尝试保持触地并逐步... | 将接触条件从 `min` 改为均值，使单腿触地也能获得部分奖励，提供连续梯度；agent 将开始尝试保持触地并逐步... | 70.55 | -71.40 | ✅ |
| 4 | 全局连续的 soft landing proxy 会让 agent 在每个 step 都能感知“接近地面且稳定”的... | 全局连续的 soft landing proxy 会让 agent 在每个 step 都能感知“接近地面且稳定”的... | 915.45 | -178.95 | ❌ |
| 5 | 引入接触传感器作为着陆奖励的触发条件，将全局 soft proxy 替换为接触门控的着陆奖励，迫使 agent 必... | 引入接触传感器作为着陆奖励的触发条件，将全局 soft proxy 替换为接触门控的着陆奖励，迫使 agent 必... | 668.35 | 30.06 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=30.056993, len=668.350000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-220.599147, 235.983745]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 280.196212 | 89.9% | 89.9% | 12.8% |
| lateral_pos_penalty | -18.143868 | -5.8% | 5.8% | 100.0% |
| progress_gated | 7.726084 | 2.5% | 4.3% | 99.3% |
| angvel_penalty | -0.127459 | -0.0% | 0.0% | 98.7% |

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
| 2 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -103.62 | -103.62 | 0.00 | 68.70 | angvel_penalty=-0.020 landing_bonus=0.016 lateral_drift_penalty=-0.039 progress_gated=0.144 | new_best |
| 3 | angvel_penalty + landing_bonus + lateral_drift_penalty + progress_gated | -71.40 | -71.40 | 0.00 | 70.55 | angvel_penalty=-0.019 landing_bonus=0.017 lateral_drift_penalty=-0.039 progress_gated=0.146 | new_best |
| 4 | angvel_penalty + landing_reward + lateral_pos_penalty + progress_gated | -178.95 | -71.40 | -107.55 | 915.45 | angvel_penalty=-0.001 landing_reward=1.111 lateral_pos_penalty=-0.025 progress_gated=0.003 | no_meaningful_improvement |
| 5 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 30.06 | 30.06 | 0.00 | 668.35 | angvel_penalty=-0.002 contact_landing_reward=2.151 lateral_pos_penalty=-0.017 progress_gated=0.026 | new_best |
