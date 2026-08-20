# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 300.000000
- current_score: 316.493351
- gap_to_target: -16.493351
- target_achievement_ratio: 105.498%

# 2. 上一轮奖励函数代码（该轮得分: 316.493351）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------- 核心观测提取 --------------------
    hull_angle     = float(obs[0])   # 身体倾角 (rad)
    hull_ang_vel   = float(obs[1])   # 身体角速度 (rad/s)
    horiz_vel      = float(obs[2])   # 前向速度 (m/s)
    vert_vel       = float(obs[3])   # 竖直速度 (m/s)

    # -------------------- 主进展奖励 --------------------
    forward_reward = 1.0 * max(0.0, horiz_vel)

    # -------------------- 稳定性门控（替换独立角度惩罚） --------------------
    # 指数衰减：倾角越大，门控值越接近 0，按比例削弱前进奖励
    # α=3.0 使得：0.3 rad → gate≈0.76, 0.5 rad → gate≈0.47, 0.8 rad → gate≈0.15
    alpha = 3.0
    stability_gate = 2.718281828 ** (-alpha * hull_angle ** 2)
    gated_forward_reward = forward_reward * stability_gate

    # -------------------- 角速度二次惩罚（保持不变） --------------------
    angvel_penalty = -0.001 * (hull_ang_vel ** 2)

    # -------------------- 竖直速度抑制（保留，后续可能调整） --------------------
    vert_threshold = 0.5
    vert_penalty = -0.1 * max(0.0, abs(vert_vel) - vert_threshold)

    # -------------------- 总奖励 --------------------
    total_reward = gated_forward_reward + angvel_penalty + vert_penalty

    components = {
        "forward_reward": forward_reward,          # 原始前进奖励（仅观测）
        "stability_gate": stability_gate,          # 门控值（用于诊断）
        "gated_forward_reward": gated_forward_reward,  # 实际计入总奖励的前进部分
        "angvel_penalty": angvel_penalty,
        "vert_penalty": vert_penalty,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 632.50 | 212.37 | ✅ |
| 2 | 将倾角惩罚改为连续门控乘在主奖励上，在每个时间步都给予姿态调整梯度，预期能降低摔倒率，提高平均步长，从而增加总前进... | 将倾角惩罚改为连续门控乘在主奖励上，在每个时间步都给予姿态调整梯度，预期能降低摔倒率，提高平均步长，从而增加总前进... | 892.95 | 316.49 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. 训练反馈
# Training Feedback

## Final-policy outcome
score=316.493351, len=892.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[314.552502, 317.895769]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 879.151491 | 46.8% | 46.8% | 100.0% |
| forward_reward | 503.443935 | 26.8% | 26.8% | 99.7% |
| gated_forward_reward | 495.486809 | 26.4% | 26.4% | 99.7% |
| angvel_penalty | -0.000652 | -0.0% | 0.0% | 99.8% |
| vert_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
环境要求一个双腿身体在不平坦地形上尽可能快、尽可能远地向前行走，同时最小化能量消耗。主体必须协调两侧髋、膝关节力矩，产生稳定的双足步态。**摔倒将导致回合提前终止**，因此保持身体平衡是隐含的刚性前提。次要是节省能量、获得平滑的关节动作，但它们绝不能以牺牲前进和平衡为代价。不应混淆“生存”与“到达终点”：虽然走过整个地形也算终止，但核心奖励必须源于持续的向前位移，而不是一次性的终点信号。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（假定）
- obs[0]: hull_angle — 身体相对竖直方向的倾斜角度，reward_usable: true（防摔倒）
- obs[1]: hull_angular_velocity — 身体角速度，reward_usable: true（姿态稳定性）
- obs[2]: horizontal_velocity — 前进/后退线速度，reward_usable: true（**主前进信号**）
- obs[3]: vertical_velocity — 上下线速度，reward_usable: true（可辅助检测跳跃或不稳定）
- obs[4]: hip1_angle — 腿1髋关节角度，reward_usable: true（姿态约束）
- obs[5]: hip1_speed — 腿1髋关节角速度，reward_usable: true（关节平滑性）
- obs[6]: knee1_angle — 腿1膝关节角度，reward_usable: true（姿态约束）
- obs[7]: knee1_speed — 腿1膝关节角速度，reward_usable: true（关节平滑性）
- obs[8]: leg1_contact — 腿1着地标志，1.0=接触，0.0=无接触，reward_usable: true（步态节律/接触）
- obs[9]: hip2_angle — 腿2髋关节角度，reward_usable: true（对称/姿态）
- obs[10]: hip2_speed — 腿2髋关节角速度，reward_usable: true（平滑）
- obs[11]: knee2_angle — 腿2膝关节角度，reward_usable: true（姿态）
- obs[12]: knee2_speed — 腿2膝关节角速度，reward_usable: true（平滑）
- obs[13]: leg2_contact — 腿2着地标志，reward_usable: true（步态节律/接触）
- obs[14..23]: lidar_0..9 — 10 路激光测距值（前方地形高度轮廓），reward_usable: 不确定，通常不直接用于简单前进奖励，可预留为高级地形感知，但基础奖励设计中不建议使用，防止引入不必要复杂度。

所有观测均可用，但是否用于奖励需依据职责。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 范围: 每维 [-1.0, 1.0]
- action_dim 0: hip_torque_leg1 — 腿1髋关节力矩
- action_dim 1: knee_torque_leg1 — 腿1膝关节力矩
- action_dim 2: hip_torque_leg2 — 腿2髋关节力矩
- action_dim 3: knee_torque_leg2 — 腿2膝关节力矩

说明：动作是连续力矩控制，无离散选项。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走完全程）
- failure-like termination: body_fallen_over（身体摔倒）
- ambiguous termination: 无
- truncation: 未在描述中提及，假定有最大步数限制（通用环境），但不作为奖励信号依据。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无法直接获得 reached_end_of_terrain 标志）
- explicit_failure_flag_available: false（同样无法直接获得摔倒标志）
- allowed_info_fields: 无（当前 step 返回 info={}）
- forbidden_or_uncertain_info_fields: 所有的 info 字段均不存在，因此不能依赖任何外部成功/失败标记。奖励函数只能基于观测和动作构建生存和前进信号。

## 7. 可用于奖励函数的信号
- position: 无直接位置观测，但可通过积分水平速度推算（但由于基础，不宜复杂，建议仅用瞬时速度）
- velocity: obs[2] horizontal_velocity（主信号），obs[3] vertical_velocity（辅助），obs[1] hull_angular_velocity（姿态稳定）
- orientation: obs[0] hull_angle（倾斜惩罚）
- contact: obs[8] leg1_contact, obs[13] leg2_contact（步态节律/双脚支撑）
- action/engine: action 四个维度的幅值（能耗惩罚），action 的前后变化量（平滑惩罚）
- other: lidar 测距阵列（obs[14:23]），不建议用于基础奖励；关节角度/角速度可用于姿态正则化。

# 6. Formula Operator Library（完整版，用于 Level 3 重建）
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



# 7. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angvel_penalty + forward_reward + vert_penalty | 212.37 | 212.37 | 0.00 | 632.50 | angle_penalty=-0.004 angvel_penalty=-0.000 forward_reward=0.358 vert_penalty=0.000 | new_best |
| 2 | angvel_penalty + forward_reward + gated_forward_reward + stability_gate + vert_penalty | 316.49 | 316.49 | 0.00 | 892.95 | angvel_penalty=-0.000 forward_reward=0.346 gated_forward_reward=0.328 stability_gate=0.921 vert_penalty=0.000 | target_solved_new_best |
