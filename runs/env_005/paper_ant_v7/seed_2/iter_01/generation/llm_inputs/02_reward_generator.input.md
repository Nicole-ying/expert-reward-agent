# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制运动任务。控制一个 3D 四足机器人（8 个扭矩关节）在保证身体高度处于健康区间、保持直立姿态的前提下，以尽可能快的速度稳定前进。核心目标是稳定的前进运动（**前进速度最大化**），维持身体高度和直立姿态是避免提前终止的必要条件，但不是任务本身的核心优化目标。不允许依赖任何官方奖励项（info 被清空，official reward masked）。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务描述明确指出“必须行走或奔跑前进”，核心目标是“稳定的前进运动”，而不是纯存活或平衡；附属的高度/姿态约束仅作为终止条件存在，并非平行多目标。典型的连续动作运动控制任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: 推断为 float32（来自 continuous locomotion）
- obs[0] (body_z)：身体重心垂直高度。reward_usable: true（可构建高度健康奖励/惩罚）
- obs[1] (quat_w)：身体姿态四元数实部 w。reward_usable: true（用于计算直立程度）
- obs[2] (quat_x)：姿态四元数虚部 x。reward_usable: true
- obs[3] (quat_y)：姿态四元数虚部 y。reward_usable: true
- obs[4] (quat_z)：姿态四元数虚部 z。reward_usable: true
- obs[5] (joint_1_angle)：第一 hip 关节角度。reward_usable: true（可用来约束关节范围、平滑动作）
- obs[6] (joint_2_angle)：第一 ankle 关节角度。reward_usable: true
- obs[7] (joint_3_angle)：第二 hip 关节角度。reward_usable: true
- obs[8] (joint_4_angle)：第二 ankle 关节角度。reward_usable: true
- obs[9] (joint_5_angle)：第三 hip 关节角度。reward_usable: true
- obs[10] (joint_6_angle)：第三 ankle 关节角度。reward_usable: true
- obs[11] (joint_7_angle)：第四 hip 关节角度。reward_usable: true
- obs[12] (joint_8_angle)：第四 ankle 关节角度。reward_usable: true
- obs[13] (body_x_velocity)：身体在世界系 x 方向的前进速度。reward_usable: **true（核心前进信号）**
- obs[14] (body_y_velocity)：身体横向速度（世界 y）。reward_usable: true（可用于惩罚侧向漂移）
- obs[15] (body_z_velocity)：身体垂直速度。reward_usable: true（可用于惩罚剧烈上下颠簸）
- obs[16] (body_roll_velocity)：滚转角速度。reward_usable: true
- obs[17] (body_pitch_velocity)：俯仰角速度。reward_usable: true
- obs[18] (body_yaw_velocity)：偏航角速度。reward_usable: true
- obs[19] (joint_1_velocity)：第一 hip 关节角速度。reward_usable: true（用于平滑或能耗惩罚）
- obs[20] (joint_2_velocity)：第一 ankle 关节角速度。reward_usable: true
- obs[21] (joint_3_velocity)：第二 hip 关节角速度。reward_usable: true
- obs[22] (joint_4_velocity)：第二 ankle 关节角速度。reward_usable: true
- obs[23] (joint_5_velocity)：第三 hip 关节角速度。reward_usable: true
- obs[24] (joint_6_velocity)：第三 ankle 关节角速度。reward_usable: true
- obs[25] (joint_7_velocity)：第四 hip 关节角速度。reward_usable: true
- obs[26] (joint_8_velocity)：第四 ankle 关节角速度。reward_usable: true

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (8,)
- 范围：[-1.0, 1.0] per joint（扭矩归一化值）
- action_dim 0 (hip_1_torque)：第一 hip 关节扭矩
- action_dim 1 (ankle_1_torque)：第一 ankle 关节扭矩
- action_dim 2 (hip_2_torque)：第二 hip 关节扭矩
- action_dim 3 (ankle_2_torque)：第二 ankle 关节扭矩
- action_dim 4 (hip_3_torque)：第三 hip 关节扭矩
- action_dim 5 (ankle_3_torque)：第三 ankle 关节扭矩
- action_dim 6 (hip_4_torque)：第四 hip 关节扭矩
- action_dim 7 (ankle_4_torque)：第四 ankle 关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无明显成功终止**。episode 如果一直保持安全状态直到被截断（truncation）则可能被视为成功完成一次稳定的前进回合。
- failure-like termination: 身体高度低于 0.2（跌倒）或高于 1.0（过度起跳）；任何状态值变为 NaN 或 inf（数值崩溃）。
- ambiguous termination: 无。
- truncation: 达到环境预设的最大仿真步数（时间限制），此时 episode 直接结束，无特殊终止标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false（terminated 信号在 step 外部返回，但 reward 函数接口无法直接获取 terminated 标志）
- allowed_info_fields: **无**（info 字典被清空，接口声明禁止使用任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部官方奖励或定位数据（明确禁止）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前状态（27维向量）
- action：当前动作（8维向量）
- next_obs：下一状态（27维向量）
- **training_progress**：仅在明确声明允许时使用（本任务 prompt 未说明，默认不应依赖）
- **info**：始终为空字典 `{}`，不得使用任何字段

禁止使用：
- original_reward（被掩码的官方奖励，接口要求禁止）
- 任何从 info 中非法获取的字段（包括但不限于 forward, ctrl, survive 等）
- 任何未声明的 obs 切片（以本卡片定义为准）

## 7. 可用于奖励函数的信号
- position: body_z（高度），关节角度（可通过与目标姿态的偏差设计奖励）
- velocity: body_x_velocity（前进速度，核心），body_y_velocity（侧向），body_z_velocity（垂直速度），各关节角速度
- orientation: body_up_z = 1 - 2*(quat_x² + quat_y²) 量化直立程度（0~1，1 为完全竖直）
- contact: 无直接接触力，本环境版本无接触信息
- action/engine: action 本身（扭矩可构成能量/平滑惩罚），action 变化量（需自行维护上次动作，但奖励函数无状态，故无法直接计算 delta；可以惩罚 action 的绝对大小）
- other: 关节角度偏离正常范围（如设定目标关节位置）可用作风格约束

## 8. 不确定或不可用的信号
- 绝对世界坐标（x_position, y_position）被禁止，无法用于计算全程位移
- 接触力、足端触地标志、地面反作用力均不可用
- 成功/失败标志不可从 info 获取，也无法从 terminated 直接传入 reward 函数
- 任何官方奖励分量（forward, ctrl, contact, survive）均不可用
- 步态周期事件（如足着地瞬间）未提供信号，不建议依赖
- 上一帧动作不可在 stateless reward 中直接使用，无法计算动作变化率

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped（4 条腿，每条腿 hip + ankle 两关节）
  actuator_type: torque_controlled (8 个独立扭矩，范围[-1,1])
  contact_structure: foot-ground contact masked（无接触力信息）
primary_objectives:
  - 最大化向前速度（body_x_velocity），维持高效前进步伐
secondary_objectives:
  - 保持身体高度在安全区间 (0.2~1.0)，避免早停
  - 维持直立姿态（body_up_z 接近 1）
  - 减小侧向漂移与垂直跳动
  - 动作平滑且节能（小扭矩、小关节速度）
main_failure_risks:
  - 因重心过低（<0.2）摔倒终止
  - 过度跳跃导致高度超过 1.0 终止
  - 关节发力过大引发数值不稳定（NaN/inf）
  - 策略陷入静止不动或原地踏步（forward velocity ~0），虽不终止但无意义
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward  
  purpose: 直接鼓励身体在世界 x 方向的前进速度  
  why_required: 是任务唯一明确的核心目标（“行走或奔跑前进”），所有其他职责均为辅助  
  usable_signals: [next_obs[13] (body_x_velocity)]  
  risks: 若权重过高可能导致策略忽略稳定性，引发早停；静止/后退策略会获得负或零奖励，需要保证梯度指向期望方向

### 10.2 条件职责 conditional_roles
- role_id: healthy_height_survival  
  purpose: 鼓励身体高度保持在安全区间中部，远离早停边界  
  condition_to_use: 当高度接近 (0.2, 1.0) 边界时给予惩罚，中部给予小奖励或零惩罚；可全程开启但需低频权重，避免与前进速度冲突  
  usable_signals: [next_obs[0] (body_z)]  
  risks: 过度奖励高度可能导致机器人专注于跳跃或维持特定高度而放弃前进

- role_id: upright_orientation  
  purpose: 惩罚身体倾斜，促使保持竖直姿态  
  condition_to_use: 通过 body_up_z 偏离 1 的程度给予平方/线性惩罚，通常全程使用  
  usable_signals: [next_obs[1:5] 计算 body_up_z]  
  risks: 过于强制可能抑制正常的步态微调，导致僵硬动作

- role_id: action_energy_penalty / smoothness  
  purpose: 惩罚过大的扭矩输出，降低能耗并鼓励平滑控制  
  condition_to_use: 始终可用，将其作为小权重正则项  
  usable_signals: [action (8维扭矩)]  
  risks: 权重过大会抑制探索，使机器人无法生成有力步伐，导致前进速度降低

- role_id: lateral_drift_penalty  
  purpose: 惩罚侧向（y）速度，抑制横向漂移  
  condition_to_use: 前进为主目标时，通常作为辅助项使用  
  usable_signals: [next_obs[14] (body_y_velocity)]  
  risks: 轻微漂移可能无害，强制为 0 可能干扰转弯（本任务不要求转弯，可接受）

- role_id: vertical_oscillation_penalty  
  purpose: 惩罚垂直速度过大，减少上下颠簸  
  condition_to_use: 可与高度保持信号共存，防止剧烈跳跃  
  usable_signals: [next_obs[15] (body_z_velocity)]  
  risks: 可能抑制正常的步态引起的轻微起伏

### 10.3 慎用/禁用职责 avoid_roles
- role_id: distance_from_start / whole_trajectory_progress  
  reason: 禁止获取 x/y 绝对坐标，无法计算累积位移，且官方定位信息被屏蔽  
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin] 均在 forbidden_info_fields 中

- role_id: contact_consistency_reward  
  reason: 本版本无接触力信息，接触模式不可知  
  forbidden_or_missing_signals: [contact forces, foot contact states] 不在 obs 空间

- role_id: any_official_reward_reproduce  
  reason: 官方奖励被掩码且明确禁止使用或重构  
  forbidden_or_missing_signals: [reward_forward, reward_ctrl, etc.] 都在 forbidden_info_fields

- role_id: goal_reaching / sparse_event  
  reason: 任务无明确目标位置，非导航/稀疏探索类型  
  reason_detail: 终止仅基于高度超限或数值问题，无成功触发标志

## 11. role_to_signal_mapping
| role_id                      | usable signals                                      | missing signals      | candidate formula operators              | notes                                                                             |
|------------------------------|-----------------------------------------------------|----------------------|------------------------------------------|-----------------------------------------------------------------------------------|
| forward_velocity_reward      | next_obs[13] (body_x_velocity)                     



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

