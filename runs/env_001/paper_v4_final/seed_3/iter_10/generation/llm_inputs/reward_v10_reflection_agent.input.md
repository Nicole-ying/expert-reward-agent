# 1. Search objective
- target_score: 200.000000
- current_score: -24.783035
- gap_to_target: 224.783035
- target_achievement_ratio: -12.392%

# 2. 上一轮奖励函数代码（该轮得分: -24.783035）
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
    w_progress = 30.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    # sigmoid with steepness 12, center 0.30
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (12.0 * (abs_angle - 0.30)) / 
                              (1.0 + 2.718281828 ** (12.0 * (abs_angle - 0.30))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.08
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. contact-based landing proxy (exponent 0.5) ----------
    mean_contact = (n_lc + n_rc) / 2.0

    k_y = 10.0
    k_vy = 8.0
    k_ang = 15.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))

    contact_landing_factor = (mean_contact * f_y * f_vy * f_ang) ** 0.5
    w_contact_land = 5.0
    contact_landing_reward = w_contact_land * contact_landing_factor

    # ---------- 6. NEW: angle-terminal-proximity penalty ----------
    # Hinge penalty activates when angle approaches a dangerous region
    # (estimated termination threshold around 1.0 rad; start warning at 0.7)
    angle_term_threshold = 0.7
    w_angle_term = 1.5
    angle_term_penalty = -w_angle_term * (max(0.0, abs_angle - angle_term_threshold) ** 2)

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + contact_landing_reward
                    + angle_term_penalty)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "contact_landing_reward": contact_landing_reward,
        "angle_term_penalty": angle_term_penalty
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
| 6 | 提升 progress 和 lateral 惩罚权重能使 agent 在保持成功着陆的前提下优化飞行效率与中心对准... | 提升 progress 和 lateral 惩罚权重能使 agent 在保持成功着陆的前提下优化飞行效率与中心对准... | 936.95 | -13.38 | ❌ |
| 7 | 将 `w_lat_pos` 回调至接近第 5 轮水平，惩罚强度降至主信号的 0.3x 以内，总奖励将恢复正向驱动，... | 将 `w_lat_pos` 回调至接近第 5 轮水平，惩罚强度降至主信号的 0.3x 以内，总奖励将恢复正向驱动，... | 1000.00 | 123.31 | ✅ |
| 8 | 骨架变化: angvel_penalty + contact_landing_reward + lateral_ | — | 69.80 | -73.51 | ❌ |
| 9 | 新增 hinge 惩罚会让 agent 在角度接近终止边界时收到直接的负梯度，主动压低角度，从而延长 episod... | 新增 hinge 惩罚会让 agent 在角度接近终止边界时收到直接的负梯度，主动压低角度，从而延长 episod... | 143.15 | -24.78 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-24.783035, len=143.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-106.555213, 19.614195]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_gated | 30.247016 | 64.5% | 75.8% | 100.0% |
| contact_landing_reward | 9.951514 | 21.2% | 21.2% | 4.6% |
| lateral_pos_penalty | -1.106193 | -2.4% | 2.4% | 100.0% |
| angvel_penalty | -0.294459 | -0.6% | 0.6% | 99.9% |
| angle_term_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
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
| 6 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -13.38 | 30.06 | -43.44 | 936.95 | angvel_penalty=-0.002 contact_landing_reward=2.420 lateral_pos_penalty=-0.046 progress_gated=0.108 | no_meaningful_improvement |
| 7 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | 123.31 | 123.31 | 0.00 | 1000.00 | angvel_penalty=-0.002 contact_landing_reward=2.166 lateral_pos_penalty=-0.008 progress_gated=0.120 | new_best |
| 8 | angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -73.51 | 123.31 | -196.82 | 69.80 | angvel_penalty=-0.009 contact_landing_reward=0.048 lateral_pos_penalty=-0.003 progress_gated=0.459 | no_meaningful_improvement |
| 9 | angle_term_penalty + angvel_penalty + contact_landing_reward + lateral_pos_penalty + progress_gated | -24.78 | 123.31 | -148.09 | 143.15 | angle_term_penalty=-0.002 angvel_penalty=-0.007 contact_landing_reward=0.051 lateral_pos_penalty=-0.003 progress_gated=0.442 | no_meaningful_improvement |
