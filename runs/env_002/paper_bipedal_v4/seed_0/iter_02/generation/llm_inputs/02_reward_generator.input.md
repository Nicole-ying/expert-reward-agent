# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
该环境要求一个双足（双支撑）智能体在崎岖地形上向前行走。  
**主目标**：尽可能快地、尽可能远地前进（距离/速度最大化）。  
**附属目标**：在前进过程中最小化能量消耗（通常表现为关节扭矩的幅度或做功）。  
智能体需要协调两条腿的髋关节与膝关节，产生稳定的双足步态；一旦身体摔倒，回合立即终止。到达地形终点也被视为一种成功终止，但智能体应当以持续前进为首要追求。

注意：不应将“到达地形终点”错误地理解为唯一的导航目标点；核心仍是在连续地形中实现高效移动，终点只是一个自然终止条件。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control  
confidence: high  
reason: 任务的核心是持续前进通过地形，前进速度（以及等效的行走距离）是核心优化指标；能量消耗为次要附属目标，非多目标权重冲突场景，因此属于典型的连续运动控制家族。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: 推测为 float32（连续数值）  
- 各维度含义与 reward 可用性：

  | 索引 | 名称 | 含义 | reward_usable |
  |------|------|------|---------------|
  | 0 | hull_angle | 主体相对于竖直方向的角度 | true |
  | 1 | hull_angular_velocity | 主体角速度 | true |
  | 2 | horizontal_velocity | 前进/后退线速度（世界坐标系水平方向，正值可能为前进） | true |
  | 3 | vertical_velocity | 上/下线速度 | true |
  | 4 | hip1_angle | 腿1髋关节角度 | true |
  | 5 | hip1_speed | 腿1髋关节角速度 | true |
  | 6 | knee1_angle | 腿1膝关节角度 | true |
  | 7 | knee1_speed | 腿1膝关节角速度 | true |
  | 8 | leg1_contact | 腿1地面接触标志（1.0=接触，0.0=未接触） | true |
  | 9 | hip2_angle | 腿2髋关节角度 | true |
  |10 | hip2_speed | 腿2髋关节角速度 | true |
  |11 | knee2_angle | 腿2膝关节角度 | true |
  |12 | knee2_speed | 腿2膝关节角速度 | true |
  |13 | leg2_contact | 腿2地面接触标志（1.0=接触，0.0=未接触） | true |
  |14..23 | lidar_0..9 | 10 个激光雷达距离测量值（前方地形扫描） | true（但使用时需谨慎防止过拟合） |

  所有观测字段均为基础物理状态，可直接用于奖励函数构造，无显式禁止使用的切片。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- 连续控制，每个维度范围 [-1.0, 1.0]  
- 动作维度含义：

| 维度 | 名称 | 含义 |
|------|------|------|
| 0 | hip_torque_leg1 | 施加在腿1髋关节上的扭矩 |
| 1 | knee_torque_leg1 | 施加在腿1膝关节上的扭矩 |
| 2 | hip_torque_leg2 | 施加在腿2髋关节上的扭矩 |
| 3 | knee_torque_leg2 | 施加在腿2膝关节上的扭矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `reached_end_of_terrain` — 智能体走完了整个地形，通常可视为成功完成行走任务。
- **failure-like termination**:  
  `body_fallen_over` — 身体摔倒（可能由 hull_angle 过大触发），属于失败终止。
- **ambiguous termination**:  
  无。
- **truncation**:  
  环境中未明确指定步数截断，但可能由外部框架设置 `max_episode_steps`（当前规格未提供），不在 reward 函数设计范围。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
  - `info` 字典为空（`{}`），无任何成功或失败的标记字段。
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 任何假设的 `info["success"]`、`info["failure"]` 等均不存在，禁止在 reward 中使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：当前时刻的观测向量（24维）
- `action`：当前时刻执行的动作向量（4维）
- `next_obs`：下一时刻的观测向量（24维）
- `info` 中明确存在的字段（当前为无，故不可用任何 info 字段）

禁止使用：
- `original_reward`（已掩码的官方奖励，不允许访问或重建）
- `official_reward` 或任何未声明的全局变量
- 未声明的 `obs` 或 `next_obs` 切片（但全部 24 维均为已声明可用）
- `training_progress` 在 prompt 未明确允许时不可使用（本环境未授权）

## 7. 可用于奖励函数的信号
基于可用观测和动作，可提取以下信号类别：

- **位置/距离相关**: 无绝对位置，但可通过 `horizontal_velocity` 的积分近似距离增量，或直接使用速度信号作为前进速率。
- **速度**:
  - `horizontal_velocity` (obs[2])：正向行进速度，可作为前进奖励核心信号。
  - `vertical_velocity` (obs[3])：可辅助检测摔倒或跳跃异常。
- **姿态/朝向**:
  - `hull_angle` (obs[0])：偏离竖直的角度，用于维持平衡或检测即将摔倒。
  - `hull_angular_velocity` (obs[1])：姿态变化速率。
- **接触**:
  - `leg1_contact` (obs[8]), `leg2_contact` (obs[13])：用于检测支撑/摆动相，可构造步态协调奖励（如鼓励交替接触、惩罚双腿同时离地等）。
- **动作/能耗**:
  - 动作向量 (action[0:4])：可直接施加动作幅度惩罚（L1/L2），以近似能量消耗。
- **其他**:
  - 激光雷达 (obs[14:23])：可反映前方地形起伏，但直接用于奖励需谨慎，通常更适用于辅助策略而非定义奖励。
  - 关节角度/角速度 (obs[4:7], 9:12)：可用于惩罚极端关节姿态、鼓励平滑运动。

## 8. 不确定或不可用的信号
- 没有明确的地形位置进度计；无法直接获得“已走水平距离”，必须从速度积分或一次 step 的位移增量构建奖励。（但 `horizontal_velocity` 足够作为每步的进度代表。）
- 无任何来自 `info` 的成功/失败标志，因此不能设计基于终点结果的条件奖励（如仅在成功时给大额奖励）。
- 没有原始的奖励或环境内部 step 计数，均不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait
control_type: continuous
morphology:
  body_type: rigid_hull_with_two_legs
  actuator_type: torque_controlled_rotational_joints (hip + knee per leg)
  contact_structure: two point-feet, each with ground contact sensing
primary_objectives:
  - maximize forward velocity / walking distance
secondary_objectives:
  - minimize energy consumption (joint torques)
main_failure_risks:
  - falling (body overturn)
  - getting stuck / not moving forward
  - unstable, jerky gait causing early fall
  - single-leg hopping to exploit speed reward without proper gait
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: forward_progress**  
  purpose: 奖励智能体向前移动，使水平速度最大化。  
  why_required: 这是任务最核心的优化目标，没有此奖励智能体将丧失前进动力。  
  usable_signals: `obs[2]` (horizontal_velocity)  
  risks: 若仅奖励速度可能导致智能体学习跳跃、摔倒前爆发冲刺等不稳定行为，需要结合稳定职责。

- **role_id: stability_and_survival**  
  purpose: 保持主体直立，防止摔倒导致的早期终止。  
  why_required: 一旦摔倒，回合结束，前进距离大幅缩短；且稳定的姿态是正常步态的基础。  
  usable_signals: `obs[0]` (hull_angle), 也可联合 `obs[1]` (hull_angular_velocity) 或 `next_obs[0]` 提前惩罚危险姿态。  
  risks: 惩罚过重可能致使智能体不敢移动（宁可原地站立），需与前进职责权衡。

### 10.2 条件职责 conditional_roles
- **role_id: energy_efficiency**  
  purpose: 惩罚过大的关节扭矩，促使智能体以能耗最低的方式行走。  
  condition_to_use: 当智能体已经学会基本前进而不频繁摔倒后，可逐步引入或加重此惩罚；初期可仅加很小的系数或不加入，避免阻碍探索。  
  usable_signals: `action[0:4]` (torques)，可使用 L2 范数或做功估算。  
  risks: 过重的能耗惩罚会压制前进步长和速度，可能导致智能体选择几乎不动的保守策略。需要与前进奖励的 scale 精心平衡。

- **role_id: gait_symmetry_or_phase



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

- target_score: 300.000
- best_score_so_far: 300.220

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| energy + forward_velocity + stability | 1 | 300.220 | 300.220 | solved |

## Previous interventions

- No structured intervention fields were available in the historical responses.

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.
