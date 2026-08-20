# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 3000.000000
- current_score: 5.504575
- gap_to_target: 2994.495425
- target_achievement_ratio: 0.183%

# 2. 上一轮奖励函数代码（该轮得分: 5.504575）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- observations ---
    body_z = obs[0]
    qx = obs[2]
    qy = obs[3]
    body_x_velocity = obs[13]
    body_y_velocity = obs[14]
    body_z_velocity = obs[15]
    body_roll_velocity = obs[16]
    body_pitch_velocity = obs[17]

    # === uprightness gate ===
    body_up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    body_up_z = max(-1.0, min(1.0, body_up_z))  # clip

    upright_threshold = 0.4
    upright_full = 0.8
    if body_up_z < upright_threshold:
        upright_gate = 0.0
    elif body_up_z < upright_full:
        upright_gate = (body_up_z - upright_threshold) / (upright_full - upright_threshold)
    else:
        upright_gate = 1.0

    # === height gate (safety multiplier on forward reward) ===
    height_low = 0.2
    height_low_margin = 0.15   # 0.2 + 0.15 = 0.35
    height_high = 0.95
    height_high_margin = 0.05  # 0.95 - 0.05 = 0.90

    if body_z < height_low:
        height_gate = 0.0
    elif body_z < height_low + height_low_margin:
        height_gate = (body_z - height_low) / height_low_margin
    elif body_z > height_high:
        height_gate = 0.0
    elif body_z > height_high - height_high_margin:
        height_gate = (height_high - body_z) / height_high_margin
    else:
        height_gate = 1.0

    gate = height_gate * upright_gate

    # === forward progress ===
    w_forward = 1.0
    forward_progress = w_forward * max(0.0, body_x_velocity) * gate

    # === lateral drift penalty (hinge) ===
    w_lat = 0.5
    lateral_drift_penalty = -w_lat * max(0.0, abs(body_y_velocity) - 0.3)

    # === roll/pitch stability hinge ===
    w_rp = 0.2
    ang_vel_margin = 1.0  # rad/s
    roll_penalty = max(0.0, abs(body_roll_velocity) - ang_vel_margin)
    pitch_penalty = max(0.0, abs(body_pitch_velocity) - ang_vel_margin)
    roll_pitch_penalty = -w_rp * (roll_penalty + pitch_penalty)

    # === descent early penalty (replaces height_boundary_hinge) ===
    # Penalty applied when falling (body_z_velocity < 0) and already below safe margin
    safe_z_margin = 0.5
    w_descend = 0.15
    descent_early_penalty = 0.0
    if body_z_velocity < 0.0 and body_z < safe_z_margin:
        # Linear penalty proportional to downward speed and proximity to low height
        fall_factor = (safe_z_margin - body_z) / safe_z_margin  # 0 when body_z = 0.5, 1 when body_z = 0
        descent_early_penalty = -w_descend * abs(body_z_velocity) * fall_factor

    total_reward = (forward_progress +
                    lateral_drift_penalty +
                    roll_pitch_penalty +
                    descent_early_penalty)

    components = {
        'forward_progress': forward_progress,
        'lateral_drift_hinge': lateral_drift_penalty,
        'roll_pitch_hinge': roll_pitch_penalty,
        'descent_early_penalty': descent_early_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 944.40 | -336.84 | ✅ |
| 2 | 骨架变化: forward_progress + lateral_drift_hinge + roll_pitc | — | 694.40 | -75.45 | ✅ |
| 3 | 骨架变化: forward_progress + height_boundary_hinge + lateral | — | 752.35 | -186.37 | ❌ |
| 4 | 骨架变化: descent_early_penalty + forward_progress + lateral | — | 968.50 | 5.50 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=5.504575, len=968.500000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[-1985.304120, 610.377687]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress | 1378.369901 | 88.5% | 88.5% | 82.7% |
| lateral_drift_hinge | -90.947874 | -5.8% | 5.8% | 49.2% |
| roll_pitch_hinge | -88.008512 | -5.7% | 5.7% | 52.6% |
| descent_early_penalty | -0.287754 | -0.0% | 0.0% | 9.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: mean_eval_reward=5.5, termination=2/20 (10%), forward_progress signed share=88.5%, descent_early_penalty dead (<0.1% share, active 9.1%)

**Component Anomalies**: descent_early_penalty dead (mean -0.0009, negligible); forward_progress dominates (88.5% signed share >70%)

**Training Dynamics**: no temporal snapshots; cannot assess growth/decay or checkpoint drift

**Signal Quality**: dead gate: descent_early_penalty condition (z_vel<0 & z<0.5) rarely active; low termination suggests missing attractor for task completion

**Evidence Confidence**: `medium`

# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
机器人需要沿着世界坐标系 x 轴（前进方向）持续、稳定地移动。主要目标是维持稳定的前进行走，而非仅仅保持直立。在实现稳定行走后，更快的向前速度应获得更高评价，但不能以牺牲稳定性为代价。身体高度必须保持在健康范围内，过低（摔倒）或过高（弹跳/飞起）都会导致回合终止。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: 连续浮点（推测 float32/64）
- obs[0]: body_z，身体垂直位置（高度），reward_usable: true
- obs[1]: quat_w，身体朝向四元数实部，reward_usable: true（需转换为 uprightness）
- obs[2]: quat_x，身体朝向四元数第一虚部，reward_usable: true（需转换为 uprightness）
- obs[3]: quat_y，身体朝向四元数第二虚部，reward_usable: true（需转换为 uprightness）
- obs[4]: quat_z，身体朝向四元数第三虚部，reward_usable: true（需转换为 uprightness）
- obs[5]: joint_1_angle，第一髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[6]: joint_2_angle，第一踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[7]: joint_3_angle，第二髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[8]: joint_4_angle，第二踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[9]: joint_5_angle，第三髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[10]: joint_6_angle，第三踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[11]: joint_7_angle，第四髋关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[12]: joint_8_angle，第四踝关节角度，reward_usable: true（可用于动作平滑/能耗）
- obs[13]: body_x_velocity，身体世界 x 轴速度（前进速度），reward_usable: true（主信号核心）
- obs[14]: body_y_velocity，身体世界 y 轴速度（侧向速度），reward_usable: true（抑制侧向漂移）
- obs[15]: body_z_velocity，身体垂直速度，reward_usable: true（监测坠落/弹跳）
- obs[16]: body_roll_velocity，翻滚角速度，reward_usable: true（稳定性指标）
- obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true（稳定性指标）
- obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true（前进方向稳定性）
- obs[19]: joint_1_velocity，第一髋关节角速度，reward_usable: true
- obs[20]: joint_2_velocity，第一踝关节角速度，reward_usable: true
- obs[21]: joint_3_velocity，第二髋关节角速度，reward_usable: true
- obs[22]: joint_4_velocity，第二踝关节角速度，reward_usable: true
- obs[23]: joint_5_velocity，第三髋关节角速度，reward_usable: true
- obs[24]: joint_6_velocity，第三踝关节角速度，reward_usable: true
- obs[25]: joint_7_velocity，第四髋关节角速度，reward_usable: true
- obs[26]: joint_8_velocity，第四踝关节角速度，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- n: 8 维连续动作
- action[0]: hip_1_torque，第一个髋关节扭矩，范围 [-1.0, 1.0]
- action[1]: ankle_1_torque，第一个踝关节扭矩，范围 [-1.0, 1.0]
- action[2]: hip_2_torque，第二个髋关节扭矩，范围 [-1.0, 1.0]
- action[3]: ankle_2_torque，第二个踝关节扭矩，范围 [-1.0, 1.0]
- action[4]: hip_3_torque，第三个髋关节扭矩，范围 [-1.0, 1.0]
- action[5]: ankle_3_torque，第三个踝关节扭矩，范围 [-1.0, 1.0]
- action[6]: hip_4_torque，第四个髋关节扭矩，范围 [-1.0, 1.0]
- action[7]: ankle_4_torque，第四个踝关节扭矩，范围 [-1.0, 1.0]

> 动作均为关节扭矩（torque），值为归一化到 [-1, 1] 后的控制信号。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明显明确成功终止。环境可能一直运行直到 truncation 或失败终止。
- failure-like termination:
  - body_height_outside_healthy_range: 身体高度超出 [0.2, 1.0]，过低（摔倒/坍塌）或过高（弹跳/飞起）。
  - state_value_outside_finite_range: 任意状态值变为 NaN 或无穷大（仿真崩溃/数值溢出）。
- ambiguous termination:
  - 无显式成功信号。如果 episode 在终止前未触发 failure，可能是在稳定行走中因步数上限 truncated。
- truncation: time_limit_reached（步数上限截断）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (failure 仅通过 terminated 标志间接体现，但不能区分失败类型)
- allowed_info_fields: []（无可用 info 字段）
- forbidden_or_uncertain_info_fields:
  - reward_forward, reward_ctrl, reward_contact, reward_survive（官方奖励分量，禁止使用）
  - x_position, y_position, distance_from_origin（绝对位置信息，禁止使用）

> 虽然没有显式 flag 可用，但可通过观测信号间接推断失败：
> - 摔倒/坍塌: body_z（obs[0]）急剧下降并接近或低于 0.2，或前一帧 body_z 明显偏高但被截断前突然更低。
> - 弹跳/飞起: body_z 突然超过 1.0 或 body_z_velocity 为正异常值。
> - 方向失控/倒地: body_up_z（由四元数计算）偏离 1.0 很严重，且 body_roll/pitch_velocity 急剧变化。
> - 仿真崩溃: 观测值出现 NaN/Inf 在终止前可能无法读取，但可提前检测。

## 7. 可用于奖励函数的信号
- position:
  - body_z (obs[0]): 身体高度，可判断是否在健康范围 [0.2, 1.0]
  - joint_angles (obs[5:13]): 关节角度，可用于动作平滑性或姿态评估
- velocity:
  - body_x_velocity (obs[13]): 前进速度（主信号核心）
  - body_y_velocity (obs[14]): 侧向速度（应抑制漂移）
  - body_z_velocity (obs[15]): 垂直速度（可检测弹跳/坠落趋势）
  - body_roll_velocity (obs[16]): 翻滚角速度（稳定性）
  - body_pitch_velocity (obs[17]): 俯仰角速度（稳定性）
  - body_yaw_velocity (obs[18]): 偏航角速度（方向稳定性）
  - joint_velocities (obs[19:27]): 关节角速度
- orientation:
  - body orientation quaternion (obs[1:5]): 可计算 body_up_z = 1 - 2*(quat_x² + quat_y²)，范围[-1, 1]，1 表示完全直立。可用于评估是否倾斜过度。
- contact: 无接触力或接触标志信号可用
- action/engine:
  - action (8D): 关节扭矩控制信号，可用于动作效率（能耗）奖励
- other:
  - derived_possible: 失败推断信号（见 5.2），通过 body_z、四元数、速度异常变化间接推测终止原因。

# 7. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_progress + lateral_drift_hinge + roll_pitch_hinge | -336.84 | -336.84 | 0.00 | 944.40 | forward_progress=0.558 lateral_drift_hinge=-0.073 roll_pitch_hinge=-0.081 | new_best |
| 2 | forward_progress + lateral_drift_hinge + roll_pitch_hinge | -75.45 | -75.45 | 0.00 | 694.40 | forward_progress=0.592 lateral_drift_hinge=-0.102 roll_pitch_hinge=-0.086 | new_best |
| 3 | forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge | -186.37 | -75.45 | -110.92 | 752.35 | forward_progress=0.595 height_boundary_hinge=-0.008 lateral_drift_hinge=-0.089 roll_pitch_hinge=-0.084 | no_meaningful_improvement |
| 4 | descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge | 5.50 | 5.50 | 0.00 | 968.50 | descent_early_penalty=-0.001 forward_progress=0.694 lateral_drift_hinge=-0.086 roll_pitch_hinge=-0.085 | new_best |
