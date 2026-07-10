# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个二维双足行走任务。智能体必须驱动一条具有髋关节和膝关节的双腿身体，在崎岖地形上尽可能远、尽可能快地向前行走，同时尽量减少能量消耗。主要目标是持续稳定前进；次要目标是降低动作能耗。一旦身体跌倒，回合立即终止；如果走到地形尽头，也会终止，这可能被视为成功完成任务。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心目标是驱动双腿身体通过不平地形持续向前移动，快速前进和能量消耗是附属要求，不存在多个等权重且冲突的核心目标。
dynamics_subtype: planar_bipedal_gait

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: 未明确给出，推断为 float32
- obs[0]: hull_angle，主体相对直立的角度，reward_usable: true
- obs[1]: hull_angular_velocity，主体角速度，reward_usable: true
- obs[2]: horizontal_velocity，前进/后退线速度，reward_usable: true
- obs[3]: vertical_velocity，上下线速度，reward_usable: true
- obs[4]: hip1_angle，腿1髋关节角度，reward_usable: true
- obs[5]: hip1_speed，腿1髋关节角速度，reward_usable: true
- obs[6]: knee1_angle，腿1膝关节角度，reward_usable: true
- obs[7]: knee1_speed，腿1膝关节角速度，reward_usable: true
- obs[8]: leg1_contact，腿1触地标志 (1.0=触地，0.0=未触地)，reward_usable: true
- obs[9]: hip2_angle，腿2髋关节角度，reward_usable: true
- obs[10]: hip2_speed，腿2髋关节角速度，reward_usable: true
- obs[11]: knee2_angle，腿2膝关节角度，reward_usable: true
- obs[12]: knee2_speed，腿2膝关节角速度，reward_usable: true
- obs[13]: leg2_contact，腿2触地标志 (1.0=触地，0.0=未触地)，reward_usable: true
- obs[14] ~ obs[23]: lidar_0..lidar_9，10 个前向激光雷达距离测量值，reward_usable: true

## 4. 动作空间 action_space
- type: Box，连续
- shape: [4]
- 范围: [-1.0, 1.0] 每关节
- action[0]: hip_torque_leg1，施加到腿1髋关节的扭矩
- action[1]: knee_torque_leg1，施加到腿1膝关节的扭矩
- action[2]: hip_torque_leg2，施加到腿2髋关节的扭矩
- action[3]: knee_torque_leg2，施加到腿2膝关节的扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（走到地形尽头，可能意味着成功完成行程）
- failure-like termination: body_fallen_over（身体跌倒）
- ambiguous termination: 无
- truncation: 未提及时间截断，从给定信息看仅有上述两种终止

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典，无任何标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 在 step 中为 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括任何潜在的 success, failure, termination_reason 等

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观测数组
- action: 当前动作数组
- next_obs: 下一时刻观测数组
- info: 当前环境中明确为 {}，不应依赖其中任何字段
- training_progress: 当前环境描述未明确允许使用，但在函数签名中存在，如确有需要需谨慎，不作为可靠信号

禁止使用：
- original_reward（官方奖励已被遮蔽）
- 任何未在 info 中明确声明的字段
- 任何未在观察说明中给出的 obs 含义以外的切片解释

## 7. 可用于奖励函数的信号
- position: 无绝对位置；可用 horizontal_velocity（obs[2]）间接衡量前进趋势
- velocity: obs[2] horizontal_velocity, obs[3] vertical_velocity, obs[1] hull_angular_velocity, 各关节速度 obs[5], obs[7], obs[10], obs[12]
- orientation: obs[0] hull_angle（身体与直立的夹角）
- contact: leg1_contact (obs[8]), leg2_contact (obs[13])，均为 0/1 触地标志
- action/engine: 动作向量 action[0..3] 可作为能量消耗惩罚的依据（如扭矩平方和）
- other: 激光雷达距离测量 obs[14..23] 可用于感知前方地形

## 8. 不确定或不可用的信号
- 绝对前进距离或位移（不能从观测中直接读取，只能从速度积分间接推断，有不稳定性）
- 显式的成功/失败标志（info 中不存在）
- 与身体跌倒相关的直接传感器信号（跌倒由内部物理引擎检测，未提供对应观测字段，可通过 hull_angle 接近极限来判断，但非精确标志）
- 地形终点位置或剩余距离（不可用）



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
- 本文件只提供通用公式算子和少量动力学类型示例，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive: `w * signal`
  - penalty: `-w * abs(error)` 或 `-w * error**2`
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：无界状态值可能支配总奖励；状态值可能被占据/刷分，而不代表任务完成。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - `x / (1 + abs(x))`
  - `1 / (1 + k * abs(error))`
  - `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或 velocity/proximity 类信号容易被刷。
- 风险：threshold 过小会导致反馈饱和或无梯度。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。

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

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * (1 / (1 + k * abs(posture_error)))`
- 使用条件：前进/接近奖励导致不健康冲刺、翻倒或失稳。
- 风险：gate 太严格会抑制探索；跳跃类任务尤其要轻。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 风险：乘积容易塌缩；单一接触或单一事件不能当成功。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

