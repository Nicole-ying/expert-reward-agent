# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 3D 四足机器人连续运动控制任务。机器人拥有 8 个扭矩控制的关节（4 条腿各含髋关节和踝关节），需要学习向前行走或跑动，同时保持身体直立并维持在健康高度范围内（0.2 ~ 1.0）。核心目标是稳定、持续地向前移动，而不是简单地保持平衡或待在原地。次要目标是动作平滑、能耗合理。不应混淆的目标：本任务没有明确的到达点或目标位置，因此不是导航到达任务；也无需抓取物体，不属于操作任务。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是使四足机器人持续向前移动（通过地形），身体姿态和高度作为约束，符合 locomotion_continuous_control 的定义。观察与动作空间均为连续，无离散决策点。

动力学子类型：dynamics_subtype: multi_legged_body_locomotion
说明：3D 四足机器人，多腿、多关节，需要协调肢体产生向前步态，同时维持身体平衡和高度，属于典型的多足身体运动控制。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [27]
- dtype: float32（推断）
- 每一维含义与奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | body_z | 机器人主身体垂直位置（高度） | true |
| 1 | quat_w | 身体姿态四元数实部 (w) | true（用于计算直立向量） |
| 2 | quat_x | 四元数分量 x | true（结合计算直立向量） |
| 3 | quat_y | 四元数分量 y | true |
| 4 | quat_z | 四元数分量 z | true |
| 5 | joint_1_angle | 第1髋关节角度 | true（用于动作/姿态罚） |
| 6 | joint_2_angle | 第1踝关节角度 | true |
| 7 | joint_3_angle | 第2髋关节角度 | true |
| 8 | joint_4_angle | 第2踝关节角度 | true |
| 9 | joint_5_angle | 第3髋关节角度 | true |
| 10 | joint_6_angle | 第3踝关节角度 | true |
| 11 | joint_7_angle | 第4髋关节角度 | true |
| 12 | joint_8_angle | 第4踝关节角度 | true |
| 13 | body_x_velocity | 身体世界坐标系下的前向速度（x 方向） | true（核心推进信号） |
| 14 | body_y_velocity | 身体侧向速度（y 方向） | true（可用于抑制侧移） |
| 15 | body_z_velocity | 身体垂直速度 | true（用于平稳性） |
| 16 | body_roll_velocity | 横滚角速度 | true |
| 17 | body_pitch_velocity | 俯仰角速度 | true |
| 18 | body_yaw_velocity | 偏航角速度 | true |
| 19 | joint_1_velocity | 第1髋关节角速度 | true |
| 20 | joint_2_velocity | 第1踝关节角速度 | true |
| 21 | joint_3_velocity | 第2髋关节角速度 | true |
| 22 | joint_4_velocity | 第2踝关节角速度 | true |
| 23 | joint_5_velocity | 第3髋关节角速度 | true |
| 24 | joint_6_velocity | 第3踝关节角速度 | true |
| 25 | joint_7_velocity | 第4髋关节角速度 | true |
| 26 | joint_8_velocity | 第4踝关节角速度 | true |

注意：没有接触力、没有世界坐标 x/y，前进进度只能通过 body_x_velocity 反映。

## 4. 动作空间 action_space
- type: Box (连续)
- shape: [8]
- 范围：[-1.0, 1.0] 每关节
- 动作维度映射：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | hip_1_torque | 第1髋关节扭矩 |
| 1 | ankle_1_torque | 第1踝关节扭矩 |
| 2 | hip_2_torque | 第2髋关节扭矩 |
| 3 | ankle_2_torque | 第2踝关节扭矩 |
| 4 | hip_3_torque | 第3髋关节扭矩 |
| 5 | ankle_3_torque | 第3踝关节扭矩 |
| 6 | hip_4_torque | 第4髋关节扭矩 |
| 7 | ankle_4_torque | 第4踝关节扭矩 |

所有动作都是连续扭矩指令，施加到对应的关节。

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 无显式成功终止。任务没有“到达目标”的概念，存活且前进是持续的。唯一正面终止是达到时间上限（truncation），表示成功存活整个回合。
- failure-like termination:
  - body_height_outside_healthy_range: 身体高度低于 0.2 或高于 1.0，分别表示跌倒或异常腾空，属于失败。
  - state_value_outside_finite_range: 任何状态出现 NaN 或 Inf，表示物理崩溃，属于失败。
- ambiguous termination: 无。
- truncation: time_limit_reached（达到最大步数），可视为存活成功，但需配合前进表现评估实际性能。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 字典为空，无 success 字段）
- explicit_failure_flag_available: false （无 failure 字段，但终止理由可由 terminated 推断）
- allowed_info_fields: [] （无可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 键都不可用，因为官方 info 被禁止。特别禁止字段：reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察（27维），可全部使用
- action: 当前动作（8维扭矩），可用
- next_obs: 下一步观察，可用于差分奖励（如正向速度增量）
- info: 字典，但此环境中明确为空，禁止使用任何字段
- training_progress: 只有 prompt 明确允许才使用，此处未提及，故不使用

禁止使用：
- original_reward（被掩码）
- official_reward（hidden）
- 任何 info 中的奖励项或位置信息（即使环境实际有，也因规则禁止使用）
- 其他未声明的环境内部变量

## 7. 可用于奖励函数的信号
- position: body_z（高度），四元数姿态（可算直立投影）
- velocity: body_x_velocity（前进速度）、body_y_velocity（侧向速度）、body_z_velocity（垂直速度）、body_roll/pitch/yaw_velocity（角速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²)（直立度量，范围 [−1,1]，１为全直立）
- contact: 无
- action/engine: 关节扭矩（action），8维连续值
- other: 可以通过 next_obs - obs 计算速度变化、高度变化等瞬时增量

## 8. 不确定或不可用的信号
- 世界坐标系下机器人的 x/y 位置：不可用，被排除在 obs 外且 info 禁止。
- 前行距离或累计位移：奖励函数是 stateless 的，无法直接获得累计前进距离，只能通过速度瞬时值推测。
- 足端接触力或触地信息：不可用，明确无此信息。
- 能耗或电机功率：可直接通过动作平方和近似，但不能获得真实扭矩做功。
- 跌倒或身体倾斜角度：可通过 body_up_z 间接估计，但不能直接得到绝对倾斜角。
- 任务成功标志：没有 success 或 failure 标签，不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped_3D
  actuator_type: torque_controlled_joints
  contact_structure: four_legs_with_feet (but contact forces not exposed)
primary_objectives:
  - maximize_forward_velocity (not cumulative distance, only instantaneous)
  - survive_episode (avoid termination due to height or NaN)
  - maintain_upright_orientation (body_up_z close to 1.0)
secondary_objectives:
  - minimize_lateral_drift (body_y_velocity near 0)
  - smooth_actions (small torques, penalize large actions)
  - minimize_vertical_oscillation (body_z_velocity near 0 ideally)
  - minimize_joint_velocities (for smooth movement)
main_failure_risks:
  - falling below min height (collapse)
  - launching above max height (over-energetic motion)
  - losing upright orientation and rolling/pitching
  - NaN/Inf due to unstable gait or large torques
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 驱使机器人产生正向速度，实现向前运动。
  why_required: 这是任务核心，没有前进激励，策略会倾向于静态存活。
  usable_signals: [body_x_velocity (obs[13])， 也可使用 next_obs[13] 或两者之差]
  risks: 单纯奖励速度可能导致高速但姿态不稳、跌倒；需与健康约束配合。

- role_id: upright_orientation
  purpose: 保持身体尽量直立，避免侧倾或前倾后仰。
  why_required: 跌倒终止条件之一是由高度判断，但姿态倾斜会很快导致高度越界；且任务描述明确要求 stay upright。
  usable_signals: [body_up_z 由 quat 计算， body_roll_velocity, body_pitch_velocity]
  risks: 过分强调直立可能抑制自然步态中的必要身体摆动，导致僵硬、前进慢。

- role_id: healthy_body_height
  purpose: 将身体高度维持在安全区间（0.2 ~ 1.0）内，且偏好接近理想高度（比如 0.5~0.6）。
  why_required: 直接关系到生存，高度超出即终止，但奖励不能只靠终止惩罚，需要密集信号引导高度。
  usable_signals: [body_z (obs[0])， next_obs[0] 变化，body_z_velocity]
  risks: 若只通过密集奖励强制高度，可能产生不自然动作；需要结合生存奖励让智能体自然调整。

- role_id: survival
  purpose: 鼓励智能体避免进入终止状态，尽可能存活全程。
  why_required: 终止即 episode 结束，存活是完成其他目标的前提。由于无显式 done flag，可以通过给每一步正的存活奖励（只要未终止）实现。
  usable_signals: [通过 terminated 标志判断是否存活，奖励函数可在 terminated=False 时加生存奖励]
  risks: 生存奖励若太高，可能削弱前进动机，导致站立不动；因此需要平衡。

### 10.2 条件职责 conditional_roles
- role_id: action_smoothness
  condition_to_use: 当策略已经学会稳定前进后，加入动作惩罚以提升动作效率和平滑性；训练初期可能妨碍探索。
  usable_signals: [action (8个扭矩)，可用平方和]
  risks: 惩罚力度过大将鼓励零动作，导致不动。

- role_id: lateral_drift_penalty
  condition_to_use: 当侧向速度明显干扰前进方向时，加入惩罚；若机器人本身行走有对称侧摆，惩罚可能错误惩罚正常步态。
  usable_signals: [body_y_velocity (obs[14])]
  risks: 可能抑制必要身体摆动，可与训练进度结合微调。

- role_id: vertical_oscillation_penalty
  condition_to_use: 若观察到身体高度周期性剧烈波动，可以奖励 body_z_velocity 接近零的倾向；但四足步行常伴随有节奏起伏，避免过度抑制。
  usable_signals: [body_z_velocity (obs[15])，body_z变化]
  risks: 抑制正常步态起伏导致行走不自然。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_force_reward
  reason: 环境未提供接触力信号，无法计算。禁止使用。

- role_id: cumulative_distance_reward
  reason: 累计前进距离需要 stateful 积分（如基于位置变化），但 reward 函数是每步 stateless，且禁止访问 x/y 位置，无法可靠计算。可以近似用速度，但非累计距离。

- role_id: explicit_success_bonus
  reason: 无成功判定信号，且任务没有到达点，不适合。

- role_id: joint_limit_penalty
  reason: 环境没有明显的关节限位越界终止（除高度外），且关节角度在观测范围内，但动态可能不需要额外惩罚；过多惩罚可能限制探索。可作为条件角色，但信号不足（不知道关节限位具体值）。暂时标记为慎用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | body_x_velocity (obs[13]), next_obs[13] 差值 | forward distance, world x position | dense_state_signal, bounded_signal, difference_reward | 可用 np.clip(vel, 0, max_speed) 或直接用 vel |
| upright_orientation | quat_w,x,y,z → body_up_z, roll/pitch velocity | contact foot positions | trigonometric_composition, penalty_from_1 | body_up_z 在 -1~1，目标 1.0，可设罚 (1 - body_up_z) |
| healthy_body_height | body_z (obs[0]), body_z_velocity | desired nominal height | quadratic_penalty, bounded_range_reward | 可设定理想高度 0.5，用 -(z - 0.5)^2；或给存活区域内的常值奖励 |
| survival | terminated flag (step-level info) | explicit done reason | step_survival_bonus | 若 not terminated，给 +1 之类的存活奖励 |
| action_smoothness | action (8D) | torque limits | action_magnitude_penalty | 用 -sum(action^2) 或 -sum(|action|) |
| lateral_drift_penalty | body_y_velocity | lateral position | quadratic_penalty | - (vel_y)^2 |
| vertical_oscillation_penalty | body_z_velocity | – | quadratic_penalty | - (vel_z)^2 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 机器人很快跌倒（高度低于 0.2） | 平均存活步数极低，body_z 快速下降 | 强化 upright_orientation 和 body_height 奖励，引入 curriculum |
| 机器人原地不动或极小速度 | body_x_velocity 均值和最大值接近 0 | 增加 forward_progress 权重，降低 action_penalty 初始值 |
| 机器人前后摆动但前进慢 | body_x_velocity 小，body_z 波动大 | 降低 vertical_oscillation 惩罚，优化步态频率 |
| 机器人高速但姿态不稳导致终止 | body_x_velocity 高但存活率低，body_up_z 波动大 | 提高 upright 和 body_height 奖励，可能限制速度上限 |
| 侧向行走或转圈 | body_y_velocity 持续非零，yaw_velocity 高 | 加入 lateral_drift_penalty 和 yaw_penalty |
| 出现 NaN 导致终止 | terminated 由 state_value_outside_finite_range 触发 | 添加动作平滑与关节速度惩罚，限制动作幅度，使用 torch.clip 保护计算 |
| 只存活但完全放弃前进（奖励 hacking） | 存活奖励过大，前进速度接近 0 | 动态调整权重，或使用 forward_progress 作为主要项



# expert_reward_context.md

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

