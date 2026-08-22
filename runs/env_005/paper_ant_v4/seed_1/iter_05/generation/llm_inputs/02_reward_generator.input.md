# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
训练一个 3D 四足机器人，通过扭矩控制 8 个关节实现稳定、快速的前向行走或奔跑，并始终保持身体高度在健康范围内 (0.2 ~ 1.0)。**核心目标**是持续前向运动，而不是单纯站立不倒；次要目标包括保持直立姿态、抑制侧向漂移、降低能耗并避免翻倒或崩溃。

## 2. 任务类型选择
**selected_route_id**: `locomotion_continuous_control`  
**confidence**: high  
**reason**: 任务核心是让四足机器人沿着一个方向（默认世界 x 轴）持续前进，附属有高度稳定、姿态保持等约束，符合“持续前进通过地形”的定义，不涉及目标点到达，也不仅仅为了平衡存活。

## 3. 观察空间 observation_space
- **type**: `Box`  
- **shape**: `[27]`  
- **dtype**: 未显式声明，通常为 `float32`（连续动力学环境）  
- 各维度含义及可用于奖励的说明：

| 索引 (index) | 名称 | 含义 | reward_usable |
|---|---|---|---|
| 0 | body_z | 身体中心垂直高度 | true（高度健康约束） |
| 1 | quat_w | 身体朝向四元数 w | true（用于计算 body_up_z） |
| 2 | quat_x | 身体朝向四元数 x | true |
| 3 | quat_y | 身体朝向四元数 y | true |
| 4 | quat_z | 身体朝向四元数 z | true |
| 5 | joint_1_angle | 第 1 髋关节角度 | true（关节姿态平滑/约束） |
| 6 | joint_2_angle | 第 1 踝关节角度 | true |
| 7 | joint_3_angle | 第 2 髋关节角度 | true |
| 8 | joint_4_angle | 第 2 踝关节角度 | true |
| 9 | joint_5_angle | 第 3 髋关节角度 | true |
| 10 | joint_6_angle | 第 3 踝关节角度 | true |
| 11 | joint_7_angle | 第 4 髋关节角度 | true |
| 12 | joint_8_angle | 第 4 踝关节角度 | true |
| 13 | body_x_velocity | 身体在世界 x 方向的前向速度 | true（前向速度奖励） |
| 14 | body_y_velocity | 身体在世界 y 方向的侧向速度 | true（侧滑惩罚） |
| 15 | body_z_velocity | 身体垂直速度 | true（跳跃/振颤抑制） |
| 16 | body_roll_velocity | 身体滚转角速度 | true（稳定性惩罚） |
| 17 | body_pitch_velocity | 身体俯仰角速度 | true |
| 18 | body_yaw_velocity | 身体偏航角速度 | true |
| 19 | joint_1_velocity | 第 1 髋关节角速度 | true |
| 20 | joint_2_velocity | 第 1 踝关节角速度 | true |
| 21 | joint_3_velocity | 第 2 髋关节角速度 | true |
| 22 | joint_4_velocity | 第 2 踝关节角速度 | true |
| 23 | joint_5_velocity | 第 3 髋关节角速度 | true |
| 24 | joint_6_velocity | 第 3 踝关节角速度 | true |
| 25 | joint_7_velocity | 第 4 髋关节角速度 | true |
| 26 | joint_8_velocity | 第 4 踝关节角速度 | true |

**重要**：环境未提供绝对世界坐标 (`x_position`, `y_position`) 和接触力信息。所有可用于奖励的信号均来自上述 27D 向量，`info` 字典为空且禁止使用任何内部拆解的奖励项。

## 4. 动作空间 action_space
- **type**: `Box` (连续)  
- **shape**: `[8]`  
- **dtype**: 通常为 `float32`，范围 `[-1.0, 1.0]` per joint  
- 每个动作维度对应一个关节的目标扭矩：

| 动作维度 | 名称 | 含义 |
|---|---|---|
| 0 | hip_1_torque | 第 1 髋关节扭矩 |
| 1 | ankle_1_torque | 第 1 踝关节扭矩 |
| 2 | hip_2_torque | 第 2 髋关节扭矩 |
| 3 | ankle_2_torque | 第 2 踝关节扭矩 |
| 4 | hip_3_torque | 第 3 髋关节扭矩 |
| 5 | ankle_3_torque | 第 3 踝关节扭矩 |
| 6 | hip_4_torque | 第 4 髋关节扭矩 |
| 7 | ankle_4_torque | 第 4 踝关节扭矩 |

扭矩范围归一化到 [-1, 1]，每个关节的加速度/力由底层模拟器转换成实际力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确成功终止条件。没有目标到达或存活时长判定。
- **failure-like termination**: 
  - `body_height_outside_healthy_range`: 身体高度 `body_z` ≤ 0.2（崩溃/倒地）或 ≥ 1.0（过度跳跃/弹射）时环境结束，属于失败。
  - `state_value_outside_finite_range`: 任何状态值变为 NaN 或 Inf，属于失败。
- **ambiguous termination**: 无。
- **truncation**: `time_limit_reached` 触发截断，不表示成功或失败。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
  观察信息中不存在任何“成功”标志。
- `explicit_failure_flag_available`: **false**  
  无法在奖励函数内部直接获取 `terminated` 或 `truncated`，只能在回合结束后从外部日志推断。任何 `info` 字段均不可用。
- `allowed_info_fields`: **无**（`info = {}`，严禁使用）
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均禁止使用，尤其禁止以下已知但被隐藏的字段：
  - `reward_forward`, `reward_ctrl`, `reward_contact`, `reward_survive`
  - `x_position`, `y_position`, `distance_from_origin`

**结论**：奖励函数 **不能** 依赖成功/失败标志来施加终止奖励或分段奖励，所有引导必须来自于观测值（`obs`）和动作（`action`）本身。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

- **允许使用**：
  - `obs`：27D 当前观测
  - `action`：8D 动作
  - `next_obs`：27D 下一步观测
  - `info` 中 **明确允许的字段**（当前无任何允许字段，故 `info` 不可用）
  - `training_progress` **仅在 prompt 明确允许时才可使用**，当前未提及，一概禁止使用。

- **禁止使用**：
  - `original_reward`（官方奖励被遮盖）
  - `official_reward` 或任何从 `info` 中拼凑出的隐含奖励项
  - 未明确声明的 `info` 字段（如 `x_position`, `y_position`）
  - 未声明的 `obs` 切片（下标 0-26 均已在观察空间中声明，不得使用超出范围的索引）
  - 依靠全局状态或跨 Episode 累积量（函数必须是无状态的）

## 7. 可用于奖励函数的信号
- **位置/姿态**：`body_z`；四元数 `quat_w, quat_x, quat_y, quat_z` → 可计算 `body_up_z = 1 - 2*(quat_x² + quat_y²)`；8 个关节角度。
- **速度**：`body_x_velocity`、`body_y_velocity`、`



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





# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: 434.480

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_progress + height_penalty + lateral_drift_penalty + upright_stability | 4 | 434.480 | 86.940 | unsolved |

## Previous interventions

- iter 2 (score=-224.850, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1——尺度修复。forward_progress 的系数过小导致其被 upright 信号完全主导，只需增大其系数即可改变相对重要性。 | 5. **selected_intervention**：仅将 forward_progress 的倍率从 2.0 改为 20.0，其他所有组件、结构、约束保持不变。
- iter 3 (score=265.020, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. **selected_level**：Level 1，因forward_progress数学形态合理但相对upright_stability的尺度不足，只需调整其系数，不改变结构。 | 5. **selected_intervention**：将forward_progress的缩放系数从20.0提高至60.0，其他所有组件保持不变。
- iter 4 (score=86.940, structure=forward_progress + height_penalty + lateral_drift_penalty + upright_stability): 4. selected_level：Level 1——高度惩罚职责合理但系数偏小（episode_sum_mean 仅 -37.7，active_rate 5.5%），仅需加大惩罚幅度即可预期改善提前终止。 | 5. selected_intervention：只修改 height_penalty 组件，将边界外惩罚系数从 20.0 提升至 60.0。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
