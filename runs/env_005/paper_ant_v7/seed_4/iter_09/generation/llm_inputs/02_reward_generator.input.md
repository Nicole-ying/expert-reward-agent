# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务要求控制一个3D四足机器人，在 8 个力矩驱动关节的作用下，持续、稳定地向前行走或奔跑，同时  
保持直立姿态与健康身体高度（不得低于 0.2 米或高于 1.0 米）。**核心目标是稳定前向移动**，而不是  
仅仅维持不倒。附属目标包括减少不必要的侧向漂移、维持平稳姿态、降低关节能耗，但这些都属于辅助  
优化，不应混淆为主目标。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务的核心是“持续前进通过平坦地形”，前向速度是主要驱动指标，高度约束仅作为安全性  
生存边界。不存在等权的多目标冲突，唯一的主线就是稳定位移，因此归入 locomotion_continuous_control。

## 3. 观察空间 observation_space
- type: Box  
- shape: (27,)  
- dtype: 连续浮点（通常为 float32）  

各维度含义及可用于奖励与否：

| index | name              | meaning（已匿名）                                  | reward_usable |
|-------|-------------------|----------------------------------------------------|---------------|
| 0     | body_z            | 主干身体垂直位置（高度）                            | true          |
| 1     | quat_w            | 身体姿态四元数实部 w                               | true          |
| 2     | quat_x            | 四元数虚部 x                                       | true          |
| 3     | quat_y            | 四元数虚部 y                                       | true          |
| 4     | quat_z            | 四元数虚部 z                                       | true          |
| 5     | joint_1_angle     | 第 1 髋关节角度                                    | true          |
| 6     | joint_2_angle     | 第 1 踝关节角度                                    | true          |
| 7     | joint_3_angle     | 第 2 髋关节角度                                    | true          |
| 8     | joint_4_angle     | 第 2 踝关节角度                                    | true          |
| 9     | joint_5_angle     | 第 3 髋关节角度                                    | true          |
| 10    | joint_6_angle     | 第 3 踝关节角度                                    | true          |
| 11    | joint_7_angle     | 第 4 髋关节角度                                    | true          |
| 12    | joint_8_angle     | 第 4 踝关节角度                                    | true          |
| 13    | body_x_velocity   | 身体在世界坐标系 x 轴（正前方）的速度               | true          |
| 14    | body_y_velocity   | 身体侧向速度                                       | true          |
| 15    | body_z_velocity   | 身体垂向速度                                       | true          |
| 16    | body_roll_velocity  | 身体绕 x 轴的角速度                              | true          |
| 17    | body_pitch_velocity | 身体绕 y 轴的角速度                              | true          |
| 18    | body_yaw_velocity   | 身体绕 z 轴的角速度                              | true          |
| 19    | joint_1_velocity  | 第 1 髋关节角速度                                  | true          |
| 20    | joint_2_velocity  | 第 1 踝关节角速度                                  | true          |
| 21    | joint_3_velocity  | 第 2 髋关节角速度                                  | true          |
| 22    | joint_4_velocity  | 第 2 踝关节角速度                                  | true          |
| 23    | joint_5_velocity  | 第 3 髋关节角速度                                  | true          |
| 24    | joint_6_velocity  | 第 3 踝关节角速度                                  | true          |
| 25    | joint_7_velocity  | 第 4 髋关节角速度                                  | true          |
| 26    | joint_8_velocity  | 第 4 踝关节角速度                                  | true          |

所有观察均可被奖励函数读取。注意：无接触力、无绝对 x/y 位置，仅可从速度推断位移趋势。

## 4. 动作空间 action_space
- type: Box  
- shape: (8,)  
- dtype: 连续（float），范围为 [-1.0, 1.0] 每个动作  

| dim | name              | 含义 (关节力矩)                                      |
|-----|-------------------|------------------------------------------------------|
| 0   | hip_1_torque      | 第 1 髋关节输出扭矩                                  |
| 1   | ankle_1_torque    | 第 1 踝关节输出扭矩                                  |
| 2   | hip_2_torque      | 第 2 髋关节输出扭矩                                  |
| 3   | ankle_2_torque    | 第 2 踝关节输出扭矩                                  |
| 4   | hip_3_torque      | 第 3 髋关节输出扭矩                                  |
| 5   | ankle_3_torque    | 第 3 踝关节输出扭矩                                  |
| 6   | hip_4_torque      | 第 4 髋关节输出扭矩                                  |
| 7   | ankle_4_torque    | 第 4 踝关节输出扭矩                                  |

动作是直接力矩命令，策略需通过这 8 个输出来同步四条腿的运动。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 时间限制截断（truncation）。如果机器人一直在健康高度、无 NaN，  
  坚持到 episode 时间上限，可视为一次成功的“持续运动”。  
- **failure-like termination**:  
  - body_height_outside_healthy_range：高度 < 0.2（跌倒）或 > 1.0（弹飞），任意一条触发即终止；  
  - state_value_outside_finite_range：任一状态值变为 NaN/inf，系统崩溃。  
- **ambiguous termination**: 无。  
- **truncation**: 仅 time_limit_reached，不属于终止条件，但作为截断信号。  
  注：truncation 本身不代表失败，但也不携带“走得多远”的绝对路径信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: [] （没有任何 info 字段可供使用）  
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等所有被明确标为 forbidden 的字段均不可用。  

因此，无法通过 info 直接获知“本次达到成功”，只能从截断或终止条件自行推断（如 truncated=true 视为存活完成，terminated=true 且高度超出范围视为失败）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs (当前观察，shape (27,))  
- action (当前动作，shape (8,))  
- next_obs (下一观察)  
- info （但该环境中 allowed_info_fields 为空，故只能使用空的 info，实质不可依赖任何 info 字段）  
- training_progress（仅在明确允许时可用，本任务无此要求，通常忽略）

禁止使用：
- original_reward（已 masked）  
- 任何在 forbidden_info_fields 中列出的字段（如 reward_forward 等）  
- 任何未在观察空间中声明的信息，例如机体绝对坐标、接触力等  

## 7. 可用于奖励函数的信号
- **position**:  
  - body_z (obs[0])：可直接作为高度信号  
  - 四元数 (obs[1:5])：可计算身体直立度 `body_up_z = 1 - 2*(quat_x² + quat_y²)`，值域约 [-1,1]，1.0 表示完全竖直  
  - 关节角度 (obs[5:13])：可用于模仿参考姿态或惩罚异常关节位置  
- **velocity**:  
  - body_x_velocity (obs[13])：前向速度，核心推进指标  
  - body_y_velocity (obs[14])：侧向速度，可惩罚侧滑  
  - body_z_velocity (obs[15])：垂向速度，可结合高度变化设计稳定性奖励  
  - 身体角速度 (obs[16:19])：可用于惩罚晃动  
  - 关节角速度 (obs[19:27])：可惩罚剧烈转动  
- **orientation**: 四元数间接提供身体倾斜信息，同 position 部分  
- **contact**: 无（无接触力、无足端压力）  
- **action/engine**:  
  - 动作 torque (action[0:8])：可用于惩罚过大力矩、平滑度（与上一动作的差异）  

## 8. 不确定或不可用的信号
- 绝对 x 坐标 / y 坐标：禁止；无法判断实际走了多远，只能通过速度积累的间接估计（但奖励函数 stateless，不能跨 step 积分）  
- 接触信息：无传感器，无法知道哪条腿着地  
- 官方奖励项：全部遮蔽，不得重建或使用  
- 成功率/失败标签：info 无相应字段  
- 环境内部动力学参数（摩擦、质量等）：不可见  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped_torso_with_four_eleven_limbs
  actuator_type: torque_controlled_joints
  contact_structure: four_leg_ground_contacts_no_force_sensor
primary_objectives:
  - generate stable forward velocity without falling
  - maintain body_z within [0.2, 1.0] and keep upright posture
secondary_objectives:
  - suppress lateral velocity and excessive joint torques
  - achieve natural-looking, energy-efficient gait
main_failure_risks:
  - body_z falls below 0.2 (leg collapse / fall)
  - body_z exceeds 1.0 (launch / extreme jump)
  - loss of upright orientation leading to imbalance
  - NaN in states due to simulation instability
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1



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





# Fresh Restart Evidence

- target_score: 2000.000
- best_score_so_far: -6.630

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| forward_reward + height_hinge_penalty + upright_reward | 1 | -6.630 | -6.630 | unsolved |
| forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | 3 | -9.560 | -337.630 | unsolved |
| forward_upright_product + height_survival_penalty + torque_penalty | 1 | -49.900 | -49.900 | unsolved |
| angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | 2 | -184.030 | -282.040 | unsolved |
| forward_upright_product + height_survival_penalty | 1 | -544.720 | -544.720 | unsolved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
