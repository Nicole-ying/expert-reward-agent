# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个3D四足机器人在平面上稳定向前移动，同时保持身体高度在 0.2 ~ 1.0 的健康范围内，并尽量维持躯干直立。  
本任务的**核心目标**是持续产生正向位移（前进速度），而不是仅仅维持站立；一旦身体高度低于 0.2（跌倒）或超过 1.0（弹飞），或出现数值溢出，episode 立即终止。  
次生目标包括保持姿态稳定、避免剧烈晃动等，但它们服务于主目标，不应替代前进速度的优先级。  
不应被混淆的目标：没有终点坐标、没有抓取物品、没有生存时间本身的奖励。

## 2. 任务类型选择
selected_route_id: **locomotion_continuous_control**  
confidence: **high**  
reason: 主目标是让四足躯干产生持续的前进位移，不涉及固定目标点、抓取操作或显式的平衡存活评判；高度/姿态约束只是健康条件与终止条件，并非独立权重相等的多目标。

动力学子类型补充：  
dynamics_subtype: **multi_legged_body_locomotion**（四足行走/奔跑，连续力矩驱动，带有落地接触的步态生成）

## 3. 观察空间 observation_space
- type: Box  
- shape: (27,)  
- dtype: 连续浮点（实际环境中为 float64 或 float32）  

各维度含义及奖励可用性：

| index | name | meaning | reward_usable |
|-------|------|---------|---------------|
| 0 | body_z | 躯干世界 Z 高度 | ✅ 可用于高度惩罚/存活判断 |
| 1 | quat_w | 躯体姿态四元数实部 | ✅ 需配合其他维度计算直立度 |
| 2 | quat_x | 四元数 x 分量 | ✅ 同上 |
| 3 | quat_y | 四元数 y 分量 | ✅ 同上 |
| 4 | quat_z | 四元数 z 分量 | ✅ 同上 |
| 5 | joint_1_angle | 髋关节1角度 | ⚠️ 通常不直接用于奖励，可用于姿态约束 |
| 6 | joint_2_angle | 踝关节1角度 | 同上 |
| 7 | joint_3_angle | 髋关节2角度 | 同上 |
| 8 | joint_4_angle | 踝关节2角度 | 同上 |
| 9 | joint_5_angle | 髋关节3角度 | 同上 |
| 10 | joint_6_angle | 踝关节3角度 | 同上 |
| 11 | joint_7_angle | 髋关节4角度 | 同上 |
| 12 | joint_8_angle | 踝关节4角度 | 同上 |
| 13 | body_x_velocity | 躯干世界 X 方向线速度（前进） | ✅ 核心前进信号 |
| 14 | body_y_velocity | 躯干世界 Y 方向线速度（侧向） | ✅ 可用于抑制侧向漂移 |
| 15 | body_z_velocity | 躯干 Z 方向线速度（垂直） | ✅ 可用于惩罚剧烈跳跃 |
| 16 | body_roll_velocity | 绕 X 轴角速度 | ✅ 可用于姿态稳定 |
| 17 | body_pitch_velocity | 绕 Y 轴角速度 | ✅ 可用于姿态稳定 |
| 18 | body_yaw_velocity | 绕 Z 轴角速度 | ✅ 可用于限制偏航 |
| 19~26 | joint_i_velocity | 对应关节的角速度 | ✅ 可用于动作平滑或关节速度惩罚 |

说明：环境明确给出 body_up_z = 1 - 2*(quat_x² + quat_y²) 可作为直立程度指标，可直接用于奖励。

## 4. 动作空间 action_space
- type: Box  
- shape: (8,)  
- continuous: true  
- 每个维度取值范围：[-1.0, 1.0]（扭矩）  

| action_dim | name | meaning |
|------------|------|---------|
| 0 | hip_1_torque | 第1条腿髋关节力矩 |
| 1 | ankle_1_torque | 第1条腿踝关节力矩 |
| 2 | hip_2_torque | 第2条腿髋关节力矩 |
| 3 | ankle_2_torque | 第2条腿踝关节力矩 |
| 4 | hip_3_torque | 第3条腿髋关节力矩 |
| 5 | ankle_3_torque | 第3条腿踝关节力矩 |
| 6 | hip_4_torque | 第4条腿髋关节力矩 |
| 7 | ankle_4_torque | 第4条腿踝关节力矩 |

奖励函数中可以使用动作值（如计算动作幅度），以抑制过大的力矩输出。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确成功终止条件（没有到达目标坐标、没有速度阈值即成功）。  
- **failure-like termination**:  
  - `body_height_outside_healthy_range`：躯干 Z 脱离 (0.2, 1.0) → 跌倒或飞起，视为失败。  
  - `state_value_outside_finite_range`：任何状态量变为 NaN 或 inf → 崩溃，视为失败。  
- **ambiguous termination**: 无。  
- **truncation**: `time_limit_reached` 超过最大步数截断 → 正常结束，不代表失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **true**（可通过 `terminated` 字段中检测到高度越界或数值异常，但无法从 info 中直接读取；通常环境返回的 `terminated` 为 True 时即失败，但奖励函数接口中 **不保证** 传入 `terminated`，需根据环境接口约定判断；本环境未提供 info 字段，禁止在奖励函数中使用 `terminated` 作为逻辑分支，只能通过观测值预判危险。）  
- allowed_info_fields: []（空列表，**禁止使用任何 info 键**）  
- forbidden_or_uncertain_info_fields:  
  - reward_forward, reward_ctrl, reward_contact, reward_survive  
  - x_position, y_position, distance_from_origin  

**重要说明**：由于 info 完全不可用，奖励函数不能依赖任何“官方奖励项”或“全局位置”。只能利用 obs 中的速度、姿态、高度等信息来构造前进、安全、平滑等信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用**：
- obs：当前观测（27维数组）
- action：当前执行的动作（8维数组）
- next_obs：下一个观测（27维数组）
- training_progress（如 prompt 明确要求可使用，否则默认禁止。本环境未明确要求使用，因此应避免依赖）
- info 中**明确允许的字段**：本环境 allowed_info_fields 为空，即**不得使用 info 中的任何字段**

**禁止使用**：
- original_reward（官方奖励被掩码，且禁止重构）
- info 中任何键（包括但不限于 reward_forward 等）
- 未声明的 obs 切片（仅可用上述已声明的 27 个维度和由四元数导出的直立度等）

## 7. 可用于奖励函数的信号
根据可用的 obs 和 action 可以构造以下信号：

- **位置/姿态**：  
  - body_z（高度）  
  - body_up_z = 1 - 2*(obs[2]² + obs[3]²)（直立投影）  
  - 各关节角度（可用于惩罚异常姿态）  
- **速度**：  
  - body_x_velocity（前进速度）  
  - body_y_velocity（侧向速度）  
  - body_z_velocity（垂直速度）  
  - body_roll_velocity, body_pitch_velocity, body_yaw_velocity（角速度）  
  - 各关节角速度  
- **接触/力**：无可用的接触力信号  
- **动作/驱动**：  
  - 8 维力矩 action（可计算动作幅度、动作变化等）

## 8. 不确定或不可用的信号
- 绝对位置（x, y）→ 不可用，无法构建“目标距离”奖励
- 接触力/接触状态 → 不可用，无法鼓励或惩罚特定足端接触
- 关节力矩的参考值（无 torque sensor）→ 不可用，只能通过 action 间接观察指令力矩
- 生存时间或累计步数 → 无法从 obs 直接读取，也不允许访问 info
- 任何来自 info 的官方奖励分量 → 禁止使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: quadruped (四足)
  actuator_type: torque-controlled joints (8 个连续力矩关节)
  contact_structure: 四足落地，与地面有硬接触（无传感器反馈）
primary_objectives:
  - 最大化躯干前进速度（body_x_velocity）
  - 保持身体高度在 (0.2, 1.0) 内（避免失败终止）
secondary_objectives:
  - 保持躯干直立（body_up_z 接近 1.0）
  - 抑制侧向漂移（body_y_velocity 接近 0）
  - 动作平滑（控制力矩大小、变化量）
main_failure_risks:
  - 跌倒（body_z < 0.2）
  - 跃起过高（body_z > 1.0）
  - 剧烈晃动导致 NaN 崩溃
  - 学到“原地踏步”只产生微弱正向速度
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: **forward_progress**
  purpose: 奖励躯干向前移动，推动机器人持续前进
  why_required: 唯一直接与任务主目标对应的信号，没有它机器人会倾向于静止或只维持平衡
  usable_signals: [body_x_velocity (obs[13])]
  risks: 如果权重过高，可能导致不顾姿态摔倒；如果线性映射，间歇前进可能得正分，需要配合姿态惩罚

- role_id: **survival_health**
  purpose: 惩罚接近危险高度的行为，或奖励高度在安全区间内
  why_required: 任务通过高度越界判定失败；奖励函数应引导策略远离危险边界，减少早期终止
  usable_signals: [body_z (obs[0])]
  risks: 如果仅惩罚边界外（但该时环境已终止），可能学不到预警；需要设计在安全区间内维持，接近边界时给惩罚

### 10.2 条件职责 conditional_roles
- role_id: **upright_stability**
  condition_to_use: 当训练初期或中后期出现侧倾/摔倒趋势时引入
  usable_signals: [body_up_z (由 quat_x, quat_y 导出，obs[2:3])， 可选补充 body_roll_velocity, body_pitch_velocity (obs[16,17])]
  risks: 过度惩罚轻微倾斜可能抑制正常步态摆动

- role_id: **lateral_drift_penalty**
  condition_to_use: 当观察到机器人明显侧向移动而非直线前进时
  usable_signals: [body_y_velocity (obs[14])]
  risks: 若机器人转向能力有限，该惩罚可能限制其纠正偏移的能力

- role_id: **action_smoothness**
  condition_to_use: 当动作剧烈抖动、耗能过大或步态不稳定时
  usable_signals: [action（8维）当前值，可与上一 action 的差值或绝对值]
  risks: 会减少探索时的随机动作幅度，可能导致过早收敛到次优步态

### 10.3 慎用/禁用职责 avoid_roles
- role_id: **goal_distance_progress**
  reason: 缺少 x,y 坐标信号，无法计算目标距离或累计位移，本职责完全不可能
  forbidden_or_missing_signals: [x_position, y_position, distance_from_origin]

- role_id: **contact_force_management**
  reason: 无可用的接触力/压力传感器信号，无法知道哪条腿触地、受力多大
  forbidden_or_missing_signals: [contact_forces, foot_contacts]

- role_id: **energy_penalty_via_torque**
  reason: 动作幅度惩罚已经可以在 conditional 中处理，且 real torque 不可观测；纯 action 惩罚即足够，无需单独角色

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | body_x_velocity (obs[13]) | – | dense_state_signal, bounded_signal, clipped_linear | 可配合 max/min 限幅防止异常速度 |
| survival_health | body_z (obs[0]) | – | bounded_signal, quadratic_penalty, barrier_function | 在 [0.25,0.9] 内给中性奖励，越界前加惩罚 |
| upright_stability | body_up_z (computed from quat), body_roll_velocity, body_pitch_velocity | – | normalized_angle_cost, absolute_penalty | 可结合角速度限制旋转速率 |
| lateral_drift_penalty | body_y_velocity (obs[14]) | – | absolute_penalty, squared_penalty | 权重宜小，避免影响转向探索 |
| action_smoothness | action (8D) and/or action delta | – | action_magnitude, action_change_penalty | 使用 L2 或 L1 范数 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 策略学习到原地踏步，仅有微弱正向速度 | body_x_velocity 长期接近 0 的正偏差，episode length 很长 | 提高 forward_progress 权重，降低生存奖励平坦区 |
| 机器人频繁摔倒（body_z < 0.2） | 高度快速下降，roll/pitch 角速度大 | 加强 upright_stability 惩罚，引入接近危险高度时的指数惩罚 |
| 机器人弹跳过高导致终止 | body_z > 1.0，伴随较大 body_z_velocity 和动作波动 | 增加 body_z



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

