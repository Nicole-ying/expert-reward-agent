# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一项 2D 双足移动任务。智能体需要控制两条腿的髋、膝共四个关节的连续力矩，驱动身体在不同地形上尽可能远、尽可能快地向前行走，同时尽量减少能量消耗。机器人必须保持直立，避免摔倒，因为摔倒将直接终止当前回合。次级目标包括：保持平稳步态、不出现过多不必要的跳跃或抖动、使用合理的关节力矩。该任务不能混淆为定点到达或生存平衡：核心驱动信号是前进速度，成功条件是到达地形尽头，而非存活时长。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务核心是在连续动作空间下持续向前通过地形，属于典型的连续控制移动任务。虽然包含能耗等次要目标，但“远、快前进”始终是第一目标，并非多目标权重相当，因此排除 multi_objective_task。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（连续值）  
- obs[0]: hull_angle – 本体与垂线的夹角（弧度），可用于惩罚倾斜，保持直立。reward_usable: true  
- obs[1]: hull_angular_velocity – 本体角速度，用于抑制旋转抖动。reward_usable: true  
- obs[2]: horizontal_velocity – 水平线速度（正向为前进），是主进度信号。reward_usable: true  
- obs[3]: vertical_velocity – 垂直线速度，可用于限制跳跃/过重落地。reward_usable: true  
- obs[4]: hip1_angle – 腿1髋关节角度。reward_usable: potentially  
- obs[5]: hip1_speed – 腿1髋关节角速度。reward_usable: potentially  
- obs[6]: knee1_angle – 腿1膝关节角度。reward_usable: potentially  
- obs[7]: knee1_speed – 腿1膝关节角速度。reward_usable: potentially  
- obs[8]: leg1_contact – 腿1是否接触地面（1.0/0.0），可用于步态协调。reward_usable: true  
- obs[9]: hip2_angle – 腿2髋关节角度。reward_usable: potentially  
- obs[10]: hip2_speed – 腿2髋关节角速度。reward_usable: potentially  
- obs[11]: knee2_angle – 腿2膝关节角度。reward_usable: potentially  
- obs[12]: knee2_speed – 腿2膝关节角速度。reward_usable: potentially  
- obs[13]: leg2_contact – 腿2是否接触地面。reward_usable: true  
- obs[14]~obs[23]: lidar_0~lidar_9 – 10个前方激光雷达测距值，反映地形轮廓，早期可不依赖。reward_usable: true，但不强制使用。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- continuous: true  
- bounds: 每个关节力矩限制在 [-1.0, 1.0]  
- action[0]: hip_torque_leg1 – 腿1髋关节力矩  
- action[1]: knee_torque_leg1 – 腿1膝关节力矩  
- action[2]: hip_torque_leg2 – 腿2髋关节力矩  
- action[3]: knee_torque_leg2 – 腿2膝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 到达地形尽头，视为成功完成。  
- failure-like termination: body_fallen_over – 机器人摔倒，视为失败。  
- ambiguous termination: 无。  
- truncation: 环境中无内部步数截断（外部训练循环可能限定最大步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （step 返回 info 为空字典）  
- explicit_failure_flag_available: false （同上）  
- allowed_info_fields: [] （空字典，无任何允许字段）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均为非法，因为环境中不提供任何信息字段。  
注意：奖励函数无法访问 `terminated` 或 `done`，必须完全基于 `obs`、`next_obs`、`action` 推断状态好坏。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观测数组）
- action（本次执行的动作数组）
- next_obs（下一观测数组，若可用）
- info：只能为空，不得假设任何键存在
- training_progress：仅当任务描述明确允许时可使用，此处无此类提示，建议不使用

禁止使用：
- original_reward（官方奖励被屏蔽，不可重现）
- 任何未在观测空间中明确声明的切片或字段
- 任何 info 字段（如 “success”、“failure”、“termination_reason” 等）
- 内部终止标志（done/terminated）

## 7. 可用于奖励函数的信号
- position: 无直接位移坐标，可通过对 horizontal_velocity 在时间上的积分近似，但奖励中通常直接使用速度信号。  
- velocity: next_obs[2]（或 obs[2]）horizontal_velocity 作为前进速度；obs[3]/next_obs[3] vertical_velocity 可用于惩罚跳跃。  
- orientation: next_obs[0] hull_angle 衡量姿态倾斜；next_obs[1] hull_angular_velocity 抑制旋转。  
- contact: next_obs[8] leg1_contact, next_obs[13] leg2_contact，用于步态协调。  
- action/engine: 本次动作 action[0:4] 可用于力矩惩罚（能耗）；也可使用动作变化量（需结合上一步动作，但接口未提供，可放弃）。  
- other: lidar 信号（next_obs[14:24] 或 obs[14:24]），可能用于感知前方地形，不强制使用。

## 8. 不确定或不可用的信号
- 真实的水平位移/位置：不可用。  
- 关节力矩限制以外的物理量（如接触力、摩擦力）：不可用。  
- 明确的成功/失败布尔标志：不可用。  
- 环境提供的原始奖励或任何额外信息字段：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: bipedal, two legs with hip and knee joints



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

