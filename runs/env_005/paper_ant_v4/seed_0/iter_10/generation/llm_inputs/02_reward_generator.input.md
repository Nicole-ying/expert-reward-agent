# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个三维四足机器人的连续控制问题。机器人拥有四条腿和八个力矩控制关节，需要向前行走或奔跑，全程保持身体高度在健康范围内，并维持基本直立的姿态。主目标是实现稳定、高效的前进运动；次要目标是抑制侧向漂移、过度抖动以及非必要的能量消耗。任务不要求到达特定位置，只关心前行速度和存活状态。不应混淆的目标包括：保持原地静止（只求平衡）或单纯避免跌倒而不前进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 核心目标是让四足机器人在连续力矩控制下持续向前运动，符合持续移动通过环境（平坦地面）的特征，无明确终止目标位置，属于经典的移动连续控制。附属的速度、姿态、能耗等要求是辅助性的，不构成多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (27,)
- dtype: float (连续)
- 各维度含义与奖励可用性：

| 索引 | 名称                        | 含义                                            | reward_usable |
|------|-----------------------------|-------------------------------------------------|---------------|
| 0    | body_z                      | 躯体垂直高度                                      | true          |
| 1    | quat_w                      | 躯体朝向四元数 w 分量                             | true（辅助计算直立） |
| 2    | quat_x                      | 躯体朝向四元数 x 分量                             | true          |
| 3    | quat_y                      | 躯体朝向四元数 y 分量                             | true          |
| 4    | quat_z                      | 躯体朝向四元数 z 分量                             | true          |
| 5    | joint_1_angle               | 第1髋关节角度                                    | true（用于平滑度量） |
| 6    | joint_2_angle               | 第1踝关节角度                                    | true          |
| 7    | joint_3_angle               | 第2髋关节角度                                    | true          |
| 8    | joint_4_angle               | 第2踝关节角度                                    | true          |
| 9    | joint_5_angle               | 第3髋关节角度                                    | true          |
| 10   | joint_6_angle               | 第3踝关节角度                                    | true          |
| 11   | joint_7_angle               | 第4髋关节角度                                    | true          |
| 12   | joint_8_angle               | 第4踝关节角度                                    | true          |
| 13   | body_x_velocity             | 世界坐标系下躯体前向速度                          | true（核心信号） |
| 14   | body_y_velocity             | 世界坐标系下躯体横向速度                          | true          |
| 15   | body_z_velocity             | 躯体垂直速度                                    | true          |
| 16   | body_roll_velocity          | 躯体滚动角速度                                  | true          |
| 17   | body_pitch_velocity         | 躯体俯仰角速度                                  | true          |
| 18   | body_yaw_velocity           | 躯体偏航角速度                                  | true          |
| 19   | joint_1_velocity            | 第1髋关节角速度                                  | true          |
| 20   | joint_2_velocity            | 第1踝关节角速度                                  | true          |
| 21   | joint_3_velocity            | 第2髋关节角速度                                  | true          |
| 22   | joint_4_velocity            | 第2踝关节角速度                                  | true          |
| 23   | joint_5_velocity            | 第3髋关节角速度                                  | true          |
| 24   | joint_6_velocity            | 第3踝关节角速度                                  | true          |
| 25   | joint_7_velocity            | 第4髋关节角速度                                  | true          |
| 26   | joint_8_velocity            | 第4踝关节角速度                                  | true          |

- body_up_z 可从 quat 推导：`1 - 2*(quat_x^2 + quat_y^2)`，可用于衡量直立程度（1.0 完全直立，0 为水平，负值为倒置）。

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- dtype: float, 连续
- 范围: [-1.0, 1.0] 每个维度
- 每个维度的含义：

| 维度 | 动作名            | 含义                       |
|------|-------------------|----------------------------|
| 0    | hip_1_torque      | 第1髋关节扭矩指令           |
| 1    | ankle_1_torque    | 第1踝关节扭矩指令           |
| 2    | hip_2_torque      | 第2髋关节扭矩指令           |
| 3    | ankle_2_torque    | 第2踝关节扭矩指令           |
| 4    | hip_3_torque      | 第3髋关节扭矩指令           |
| 5    | ankle_3_torque    | 第3踝关节扭矩指令           |
| 6    | hip_4_torque      | 第4髋关节扭矩指令           |
| 7    | ankle_4_torque    | 第4踝关节扭矩指令           |

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确的成功终止条件；存活至截断可视为隐式成功。
- failure-like termination:
  - body_height_outside_healthy_range: body_z 落在 (0.2, 1.0) 之外（倒地或过度腾空）。
  - state_value_outside_finite_range: 任何状态值变为 NaN 或 Inf。
- ambiguous termination: 无。
- truncation: 达到预设最大时间步数（time_limit_reached），视为正常截断，不应被惩罚。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 中无任何成功标志）。
- explicit_failure_flag_available: false（终止原因不通过 info 暴露）。
- allowed_info_fields: []（空的，禁止使用任何 info 字段）。
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等均被屏蔽，不可用。
- 备注：终止条件只能在环境交互层面感知（如训练循环），`compute_reward` 内部无法直接获取 `terminated` 标志（除非通过 `next_obs` 中的无效值推断，但不建议依赖）。奖励设计必须依赖观测信号（如当前身体高度、速度）来预测并避免失败区域。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观测（27维数组）
- `action`：当前执行的动作（8维数组）
- `next_obs`：下一时刻观测（27维数组）
- `info` 中明确允许的字段（本环境中无允许字段，禁止访问任何键）
- `training_progress`：仅在 prompt 明确允许时可用（本任务未提及，因此禁止使用）

禁止使用：
- `original_reward`（已被屏蔽）
- `official_reward` 或任何隐藏在 info 中的原有奖励项
- `info` 的任何字段（包括可能存在的 success, failure, termination_reason 等）
- `x_position`, `y_position`, `distance_from_origin` 等被屏蔽的位置信息
- 未声明的观测维度切片（例如，不得假设未来观测中存在绝对位置）

## 7. 可用于奖励函数的信号
- position:
  - body_z：身体高度，可构建高度安全奖励/惩罚。
  - quat：可计算 body_up_z，反映直立程度。
- velocity:
  - body_x_velocity：前进速度（核心正奖励信号）。
  - body_y_velocity：横向速度（可用作方向保持惩罚）。
  - body_z_velocity：垂直速度（可抑制跳跃倾向）。
  - body_roll/pitch/yaw_velocity：角速度（稳定性惩罚）。
- joint state:
  - 所有关节角度和角速度（可用于动作平滑性、关节限位惩罚）。
- action/engine:
  - action（扭矩向量）：可构建能耗惩罚或动作幅度正则化。
- other:
  - 没有接触力、足底触地状态等信号。

## 8. 不确定或不可用的信号
- 绝对位置（世界坐标 x, y）：不可用，无法计算距离进度。
- 接触力或接触标志：不可用，无法奖励正确的着地模式。
- 官方奖励项（forward, ctrl, contact, survive）：被屏蔽，禁止使用。
- 终止标志、成功/失败标签：info 不可访问，不能作为奖励依据。
- 环境内部状态（如质心位置、关节力矩限制等）：未暴露。
- 训练进度参数：未允许使用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 三维四足躯体，8个旋转关节（髋+踝）
  actuator_type: 力矩直接控制，无齿轮或位置伺服模拟
  contact_structure: 足端与地面接触（接触力不可观测）
primary_objectives:
  - 持续向前高速移动（最大化 body_x_velocity 正方向分量）
  - 保持躯体直立（body



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
- best_score_so_far: 203.900

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty | 3 | 203.900 | 52.690 | unsolved |
| angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 4 | 26.070 | 26.070 | unsolved |
| ang_penalty + forward_reward + lat_penalty | 2 | -48.960 | -172.450 | unsolved |

## Previous interventions

- iter 3 (score=24.080, structure=angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward): 4. **selected_level**：Level 2，因为必要信号可达但现有门控阈值（0.35）未能及时衰减速度奖励，属于“约束在无关阶段妨碍探索”变体——需要在高度下降初期就施加阶段相关的速度抑制，故选择global_to_local_gate思路修改velocity_reward中的门控。 | 5. **selected_intervention**：仅修改velocity_reward组件内的health_factor低高度阈值，将线性衰减区间从[0.2, 0.35]扩大到[0.2, 0.5]，使身体高度低于0.5时前向速度奖励即开始成比例衰减。
- iter 4 (score=26.070, structure=angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward): 4. **selected_level**: Level 1 — the height_penalty component has a reasonable mathematical form (quadratic, signed), but its extreme coefficient (10) makes it too strong relative to other components, poisoning the rewar | 5. **selected_intervention**: Reduce the height_penalty coefficient from 10.0 to 2.0, keeping the quadratic form and all other components unchanged.
- iter 6 (score=-172.450, structure=ang_penalty + forward_reward + lat_penalty): 4. selected_level：Level 2（结构变换），将奖励计算从 obs 切换到 next_obs，以纠正信用分配错误；这符合“修正观测源”这类最小但可验证的数学结构修改。 | 5. selected_intervention：唯一目标组件 forward_reward。修改其计算：用 next_obs 替换 obs 来获取 body_x_vel 以及计算 combined_gate（高度和直立），其他结构、系数和惩罚项保持不变。
- iter 7 (score=203.900, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. selected_level：Level 3 重建主骨架——当前骨架族（forward_reward + lat/ang penalty + 门控）连续两轮未刷新best、历史最佳仅26（远低于target 2000的25%），且已做Level 1尺度调整无效，满足重建条件。 | 5. selected_intervention：以iter4的组分框架为基础，将前进奖励从线性改为平方项（仅对正前向速度）以提供更强梯度，移除乘性健康/直立门控，改用温和的加性高度/直立惩罚维持存活，保留侧向与角速度惩罚。
- iter 8 (score=147.710, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. **selected_level**：Level 2，因为直立惩罚的数学形态导致其只在倾斜≥60°后起作用，证据表明agent在该阶段已难以恢复，需要连续惩罚来提供更密集的直立梯度。 | 5. **selected_intervention**：仅修改`upright_penalty`，从`10.0 * max(0.0, 0.5 - body_up_z)`变为`5.0 * ((1.0 - body_up_z) ** 2)`，使任何偏离完全直立的状态都产生连续惩罚，强度随倾角平方增长。
- iter 9 (score=52.690, structure=ang_penalty + forward_reward + height_penalty + lat_penalty + upright_penalty): 4. `selected_level`：Level 2，因为同一骨架已连续两轮未刷新best（iter 7→8下降），且目标完成率仅7.4%；高度惩罚尺度太弱，改变其数学形态以形成有效保护。 | 5. `selected_intervention`：唯一目标组件为`height_penalty`，将低高度惩罚由线性改为二次且阈值从0.3提前到0.4，系数从20提升到100，使其在接近摔倒时急剧增大。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
