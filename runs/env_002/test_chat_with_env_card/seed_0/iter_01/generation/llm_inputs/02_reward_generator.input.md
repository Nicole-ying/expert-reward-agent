# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主目标是控制一个双足身体在崎岖地形上尽可能远、尽可能快地向前行走，同时最小化能量消耗。次目标是保持身体平衡不摔倒。不应混淆的目标是：任务不是到达某个精确目标点，也不是跳跃或奔跑，而是持续稳定的双足步态前进。能量最小化是附属优化目标，不是核心任务目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是持续前进通过地形，属于典型的连续控制步态任务。附属的能耗优化和速度优化不改变主任务性质。

dynamics_subtype: planar_bipedal_gait
reason: 双足、平面运动、髋关节和膝关节协调产生步态，典型平面双足步态动力学。

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32 (推测)
- obs[0]: hull_angle，身体相对于竖直方向的角度，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前后线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2地面接触标志(1.0=接触,0.0=无接触)，reward_usable: true
- obs[14..23]: lidar_0..9，10个LIDAR测距仪沿前方地形测量距离，reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- dtype: float32 (推测)
- bounds: [-1.0, 1.0] per dimension
- action[0]: hip_torque_leg1，腿1髋关节施加的扭矩
- action[1]: knee_torque_leg1，腿1膝关节施加的扭矩
- action[2]: hip_torque_leg2，腿2髋关节施加的扭矩
- action[3]: knee_torque_leg2，腿2膝关节施加的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain — 成功到达地形终点，属于成功终止
- failure-like termination: body_fallen_over — 身体摔倒，属于失败终止
- ambiguous termination: 无
- truncation: 无（truncated=False 固定返回）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无 success 字段）
- explicit_failure_flag_available: false（info 字典为空，无 failure 字段）
- allowed_info_fields: 无（info 字典为空 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为 info 为空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前步的观测 (24维)
- action: 当前步的动作 (4维)
- next_obs: 下一步的观测 (24维)
- info: 当前为空，但未来可能扩展，需谨慎
- training_progress: 仅当 prompt 明确允许时才使用

禁止使用：
- original_reward: 官方奖励已屏蔽，禁止使用
- 未声明的 info 字段: info 为空，无可用字段
- 未声明的 obs 切片: 仅使用上述定义的 24 维

## 7. 可用于奖励函数的信号
- position: 无直接位置信号，但可通过 horizontal_velocity 积分或 LIDAR 间接推断
- velocity: horizontal_velocity (obs[2]), vertical_velocity (obs[3]), hull_angular_velocity (obs[1]), joint speeds (obs[5], obs[7], obs[10], obs[12])
- orientation: hull_angle (obs[0]), joint angles (obs[4], obs[6], obs[9], obs[11])
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])
- action/engine: action 本身 (4维扭矩)
- other: LIDAR 测距 (obs[14..23])

## 8. 不确定或不可用的信号
- 绝对位置 (x, y): 不可用，无直接位置观测
- 地形高度/坡度: 不可直接获取，但可通过 LIDAR 间接推断
- 能耗/功率: 不可直接获取，但可通过 action 平方和近似
- 步态周期/相位: 不可直接获取，需从接触信号推断
- 成功/失败标志: info 为空，无 explicit 标志
- 步数/时间: 不可用，无时间步计数

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal (two-legged)
  actuator_type: torque-controlled hip and knee joints (4 actuators)
  contact_structure: two feet with binary ground contact sensors
primary_objectives:
  - 向前行走尽可能远 (最大化前进距离)
  - 保持身体平衡不摔倒
secondary_objectives:
  - 尽可能快地行走 (最大化前进速度)
  - 最小化能量消耗 (最小化动作幅度/扭矩)
main_failure_risks:
  - 身体倾斜过大导致摔倒 (body_fallen_over)
  - 步态不稳定导致无法持续前进
  - 关节角度超出安全范围
  - 地形崎岖导致绊倒或失去平衡
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励身体向前移动，实现行走目标
  why_required: 核心任务目标是向前行走，必须提供正向激励
  usable_signals: [horizontal_velocity (obs[2])]
  risks: 可能鼓励摔倒式前进，需配合平衡约束

- role_id: balance_maintenance
  purpose: 保持身体直立不摔倒
  why_required: 摔倒即终止，必须惩罚倾斜或角速度过大
  usable_signals: [hull_angle (obs[0]), hull_angular_velocity (obs[1])]
  risks: 过度约束可能导致动作僵硬

### 10.2 条件职责 conditional_roles
- role_id: energy_efficiency
  condition_to_use: 当训练中期步态基本稳定后，加入能耗惩罚以优化动作平滑度
  usable_signals: [action (4维扭矩)]
  risks: 过早加入可能导致 agent 不敢动作，无法学习行走

- role_id: gait_rhythm
  condition_to_use: 当 agent 已能行走但步态不协调时，通过接触信号鼓励交替步态
  usable_signals: [leg1_contact (obs[8]), leg2_contact (obs[13])]
  risks: 过度约束可能限制自然步态多样性

- role_id: terrain_adaptation
  condition_to_use: 当 LIDAR 显示地形崎岖时，鼓励根据地形调整步态
  usable_signals: [lidar_0..9 (obs[14..23])]
  risks: 信号维度高，可能增加学习难度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: speed_maximization
  reason: 过度追求速度可能导致步态不稳定、摔倒风险增加，且与能耗最小化冲突
  forbidden_or_missing_signals: [无直接速度上限信号]

- role_id: joint_angle_limits
  reason: 环境物理引擎可能已内置关节限位，重复惩罚可能导致双重惩罚
  forbidden_or_missing_signals: [无关节限位观测]

- role_id: contact_force_minimization
  reason: 无接触力观测信号，无法实现
  forbidden_or_missing_signals: [contact_force 信号不存在]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_velocity (obs[2]) | 无 | dense_state_signal, clipped_linear | 直接使用水平速度作为正向奖励 |
| balance_maintenance | hull_angle (obs[0]), hull_angular_velocity (obs[1]) | 无 | bounded_signal, quadratic_penalty | 惩罚角度偏离竖直和角速度过大 |
| energy_efficiency | action (4维) | 无 | quadratic_penalty, l2_norm | 惩罚动作平方和 |
| gait_rhythm | leg1_contact (obs[8]), leg2_contact (obs[13]) | 步态相位 | binary_contact_alternation | 鼓励交替接触 |
| terrain_adaptation | lidar_0..9 (obs[14..23]) | 地形高度 | distance_to_obstacle_penalty | 根据 LIDAR 调整步态 |
| speed_maximization | horizontal_velocity (obs[2]) | 速度上限 | 无 | 禁用 |
| joint_angle_limits | 无 | joint_limit_observations | 无 | 禁用 |
| contact_force_minimization | 无 | contact_force | 无 | 禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 摔倒频繁 | 终止率过高，hull_angle 绝对值过大 | 增加 balance_maintenance 权重，或加入 hull_angle 的 soft 约束 |
| 原地不动/后退 | horizontal_velocity 均值接近 0 或负值 | 增加 forward_progress 奖励幅度，或加入最小速度阈值 |
| 步态不对称 | 一条腿接触时间远多于另一条 | 加入 gait_rhythm 条件职责，鼓励交替步态 |
| 动作抖动/高频振荡 | action 变化率过大，能耗过高 | 加入 energy_efficiency 惩罚，或对 action 差分施加惩罚 |
| 地形适应不良 | 在 LIDAR 显示障碍时摔倒或减速 | 加入 terrain_adaptation 条件职责，利用 LIDAR 信号 |
| 陷入局部最优（如跳跃式前进） | 步态异常，vertical_velocity 波动大 | 加入 gait_rhythm 或限制垂直速度 |
| 过早终止（未到终点） | 平均 episode 长度过短 | 检查 balance_maintenance 是否过弱，或 forward_progress 是否过强导致冒险 |



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

