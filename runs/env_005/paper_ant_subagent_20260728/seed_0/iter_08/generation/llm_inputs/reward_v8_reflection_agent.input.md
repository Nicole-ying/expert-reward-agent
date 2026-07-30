# Duplicate reward retry
The previous generation duplicated iter 6 (runs/env_005/paper_ant_subagent_20260728/seed_0/iter_06/generation/reward_v6.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (continuous, 1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x ** 2 + quat_y ** 2)

    # ---- height gate: hat-shaped, 1.0 in safe zone, decays to 0 near limits ----
    z_low       = 0.25
    z_high      = 0.95
    z_safe_low  = 0.35
    z_safe_high = 0.85

    low_factor = (body_z - z_low) / (z_safe_low - z_low)
    low_factor = max(0.0, min(1.0, low_factor))

    high_factor = (z_high - body_z) / (z_high - z_safe_high)
    high_factor = max(0.0, min(1.0, high_factor))

    height_gate = low_factor * high_factor

    # ---- height-gated forward progress (main signal) ----
    w_fwd = 1.0
    forward_progress = w_fwd * v_x * height_gate

    # ---- soft upright bonus (no gating of forward) ----
    w_up = 0.1
    upright_bonus = w_up * max(0.0, up_z)

    # ---- lateral stability (hinge: allows mild drift) ----
    lat_threshold = 0.5
    lat_err = max(0.0, abs(v_y) - lat_threshold)
    w_lat = 0.1
    lateral_penalty = -w_lat * (lat_err ** 2)

    # ---- action smoothness (light regularizer) ----
    w_act =
```

# 1. Search objective
- target_score: 2000.000000
- current_score: -277.904767
- gap_to_target: 2277.904767
- target_achievement_ratio: -13.895%

# 2. 上一轮奖励函数代码（该轮得分: -277.904767）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- signal extraction ----
    body_z   = obs[0]
    quat_x   = obs[2]
    quat_y   = obs[3]
    v_x      = obs[13]
    v_y      = obs[14]

    # ---- upright measure (continuous, 1.0 when perfectly upright) ----
    up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)

    # ---- upright gate (soft gating of forward progress) ----
    gate_lower = 0.3
    gate = (up_z - gate_lower) / (1.0 - gate_lower)
    gate = max(0.0, min(1.0, gate))

    # ---- gated forward progress (main progress signal) ----
    w_fwd = 1.0
    gated_forward = w_fwd * v_x * gate

    # ---- upright bonus (direct posture reward, increased weight) ----
    w_up = 0.5
    upright_bonus = w_up * up_z

    # ---- body height safety (soft hinge quadratic) ----
    z_low_safe  = 0.3
    z_high_safe = 0.8
    w_h = 0.5
    low_hinge  = max(0.0, z_low_safe - body_z)
    high_hinge = max(0.0, body_z - z_high_safe)
    height_penalty = -w_h * (low_hinge**2 + high_hinge**2)

    # ---- lateral stability (mild quadratic, calibrated to ~0.1 per-step) ----
    w_lat = 0.3
    lateral_penalty = -w_lat * (v_y ** 2)

    # ---- action smoothing (light energy regularizer) ----
    w_act = 0.005
    action_penalty = -w_act * sum(a**2 for a in action) / len(action)

    # ---- total reward ----
    total_reward = (gated_forward + upright_bonus +
                    height_penalty + lateral_penalty +
                    action_penalty)

    components = {
        "gated_forward":      gated_forward,
        "upright_bonus":      upright_bonus,
        "height_penalty":     height_penalty,
        "lateral_penalty":    lateral_penalty,
        "action_penalty":     action_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 11.80 | 0.72 | ✅ |
| 2 | 移除 health_gate 释放 forward 信号 + 将 upright 改为连续温和二次惩罚 → age... | 移除 health_gate 释放 forward 信号 + 将 upright 改为连续温和二次惩罚 → age... | 503.55 | -383.09 | ❌ |
| 3 | 将前进收益门控在直立姿态上，迫使 agent 必须先保持直立才能获利，能打破两大项对冲的局面，使 forward ... | 将前进收益门控在直立姿态上，迫使 agent 必须先保持直立才能获利，能打破两大项对冲的局面，使 forward ... | 981.50 | 1839.71 | ✅ |
| 4 | 将 lateral_penalty 改为 hinge 形式并降低平均惩罚量，可释放约130~150分净收益，使 s... | 将 lateral_penalty 改为 hinge 形式并降低平均惩罚量，可释放约130~150分净收益，使 s... | 585.90 | -591.52 | ❌ |
| 5 | 添加 roll/pitch 角速度的 hinge 惩罚将抑制快速翻滚，减少摔倒并恢复前进能力。 | 添加 roll/pitch 角速度的 hinge 惩罚将抑制快速翻滚，减少摔倒并恢复前进能力。 | 700.30 | -73.65 | ❌ |
| 6 | 恢复至 iter3 的无阈值二次惩罚骨架，并提升 upright_bonus 权重，能稳定 gate 学习曲线，避... | 恢复至 iter3 的无阈值二次惩罚骨架，并提升 upright_bonus 权重，能稳定 gate 学习曲线，避... | 837.25 | -442.53 | ➖ |
| 7 | 通过将 `w_lat` 从 0.05 提升到 0.30，使每步横向惩罚恢复至约 -0.10，agent 会被迫保持... | 通过将 `w_lat` 从 0.05 提升到 0.30，使每步横向惩罚恢复至约 -0.10，agent 会被迫保持... | 986.65 | -277.90 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-277.904767, len=986.650000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-367.212175, -196.249790]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| gated_forward | 1030.828718 | 65.7% | 65.9% | 96.8% |
| upright_bonus | 459.124095 | 29.3% | 31.1% | 100.0% |
| lateral_penalty | -44.013497 | -2.8% | 2.8% | 97.9% |
| action_penalty | -2.882555 | -0.2% | 0.2% | 100.0% |
| height_penalty | -0.071885 | -0.0% | 0.0% | 5.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个3D四足机器人向前稳定行走/奔跑。核心目标是产生持续的前向速度，同时保持身体高度在安全范围（0.2 ~ 1.0）内不摔倒。次要目标包括维持直立姿态、减少侧向漂移、控制能耗和动作平滑。任务 **不要求** 到达某个指定位置，仅要求长期存活并向前移动。不能混淆为“仅站立不动”或“最小化能量消耗”，前进是刚性主目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 连续浮点数（具体精度由环境决定）
- obs[0] (body_z): 身体高度，reward_usable: true，可用作安全高度监控
- obs[1] (quat_w): 身体姿态四元数实部，reward_usable: true，参与直立度计算
- obs[2] (quat_x): 四元数虚部 x，reward_usable: true
- obs[3] (quat_y): 四元数虚部 y，reward_usable: true
- obs[4] (quat_z): 四元数虚部 z，reward_usable: true
- obs[5] (joint_1_angle): 髋关节1角度，reward_usable: true（可做动作平滑或参考姿态）
- obs[6] (joint_2_angle): 踝关节1角度，reward_usable: true
- obs[7] (joint_3_angle): 髋关节2角度，reward_usable: true
- obs[8] (joint_4_angle): 踝关节2角度，reward_usable: true
- obs[9] (joint_5_angle): 髋关节3角度，reward_usable: true
- obs[10] (joint_6_angle): 踝关节3角度，reward_usable: true
- obs[11] (joint_7_angle): 髋关节4角度，reward_usable: true
- obs[12] (joint_8_angle): 踝关节4角度，reward_usable: true
- obs[13] (body_x_velocity): 世界x轴（前向）速度，reward_usable: true，**主前向奖励信号**
- obs[14] (body_y_velocity): 世界y轴（侧向）速度，reward_usable: true，可惩罚侧向
- obs[15] (body_z_velocity): 垂直速度，reward_usable: true，可惩罚剧烈上下起伏
- obs[16] (body_roll_velocity): 滚转角速度，reward_usable: true，用于稳定性惩罚
- obs[17] (body_pitch_velocity): 俯仰角速度，reward_usable: true
- obs[18] (body_yaw_velocity): 偏航角速度，reward_usable: true，转弯惩罚
- obs[19] (joint_1_velocity): 关节1角速度，reward_usable: true（动作平滑/能耗）
- obs[20] (joint_2_velocity): 关节2角速度，reward_usable: true
- obs[21] (joint_3_velocity): 关节3角速度，reward_usable: true
- obs[22] (joint_4_velocity): 关节4角速度，reward_usable: true
- obs[23] (joint_5_velocity): 关节5角速度，reward_usable: true
- obs[24] (joint_6_velocity): 关节6角速度，reward_usable: true
- obs[25] (joint_7_velocity): 关节7角速度，reward_usable: true
- obs[26] (joint_8_velocity): 关节8角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- 连续动作，每个维度范围 [[-1.0, 1.0]]
- action_dim 0: hip_1_torque — 第一髋关节扭矩
- action_dim 1: ankle_1_torque — 第一踝关节扭矩
- action_dim 2: hip_2_torque — 第二髋关节扭矩
- action_dim 3: ankle_2_torque — 第二踝关节扭矩
- action_dim 4: hip_3_torque — 第三髋关节扭矩
- action_dim 5: ankle_3_torque — 第三踝关节扭矩
- action_dim 6: hip_4_torque — 第四髋关节扭矩
- action_dim 7: ankle_4_torque — 第四踝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止标志；可默认为“在时间限制（truncation）内始终保持健康姿态”视为一次成功完整运行。
- **failure-like termination**:  
  - body_height_outside_healthy_range：身体高度 z ≤ 0.2（摔倒）或 z ≥ 1.0（过度跃起）。  
  - state_value_outside_finite_range：任何状态值变为 NaN 或 inf，通常代表物理崩溃。  
  两类均直接终止回合，属于硬失败。
- **ambiguous termination**: 无。
- **truncation**: time_limit_reached（达到最大步数），表示存活完全程。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** 
- explicit_failure_flag_available: **false** （`info` 字段为空，不能直接获得终止原因，仅能从环境返回的 `terminated` 或 `truncated` 在 RL 循环中判断，但奖励函数接口不提供这些标志）
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin

## 7. 可用于奖励函数的信号
- **位置相关**：身体高度 `body_z`（obs[0]）；身体姿态四元数 `quat_w,x,y,z`（obs[1:5]），可计算 body_up_z。关节角度（obs[5:13]）可构造姿态正则化或对称性惩罚。
- **速度相关**：前向速度 `body_x_velocity`（obs[13]）——直接前进奖励；侧向速度 `body_y_velocity`（obs[14]）——侧向漂移惩罚；垂直速度 `body_z_velocity`（obs[15]）——起伏惩罚；角速度 `body_roll/pitch/yaw_vel`（obs[16:19]）——稳定性和转向惩罚；关节角速度（obs[19:27]）——动作平滑/能耗。
- **动作/执行器**：`action`（8维扭矩）可用于计算力矩大小、变化量。
- **其他**：训练进度（若环境描述明确需要，但此处未强调，谨慎使用）。

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | 0.72 | 0.72 | 0.00 | 11.80 | action_penalty=-0.003 forward=0.222 height_penalty=-0.032 lateral_penalty=-0.097 upright_penalty=-5.673 | new_best |
| 2 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | -383.09 | 0.72 | -383.81 | 503.55 | action_penalty=-0.003 forward=0.372 height_penalty=-0.003 lateral_penalty=-0.117 upright_penalty=-0.758 | no_meaningful_improvement |
| 3 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | 1839.71 | 1839.71 | 0.00 | 981.50 | action_penalty=-0.003 gated_forward=1.150 height_penalty=-0.000 lateral_penalty=-0.103 upright_bonus=0.095 | new_best |
| 4 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -591.52 | 1839.71 | -2431.23 | 585.90 | action_penalty=-0.003 gated_forward=0.141 height_penalty=-0.001 lateral_penalty=-0.015 upright_bonus=-0.011 | no_meaningful_improvement |
| 5 | action_penalty + ang_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -73.65 | 1839.71 | -1913.36 | 700.30 | action_penalty=-0.003 ang_penalty=-0.081 gated_forward=0.647 height_penalty=-0.000 lateral_penalty=-0.024 | no_meaningful_improvement |
| 6 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -442.53 | 1839.71 | -2282.23 | 837.25 | action_penalty=-0.003 gated_forward=0.428 height_penalty=-0.000 lateral_penalty=-0.020 upright_bonus=0.114 | unsolved_high_achievement_continue_from_best |
| 7 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -277.90 | 1839.71 | -2117.61 | 986.65 | action_penalty=-0.003 gated_forward=0.361 height_penalty=-0.000 lateral_penalty=-0.060 upright_bonus=0.194 | no_meaningful_improvement |
