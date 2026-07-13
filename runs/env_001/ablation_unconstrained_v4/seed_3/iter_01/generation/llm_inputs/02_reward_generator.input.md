# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个二维飞行器的轨迹优化问题。主体从视口上方中央附近以随机初始受力出发，需要在最短时间内到达并稳定在中央目标着陆垫上，同时尽量少用引擎推力。智能体必须学会向目标靠近，减小水平和垂直速度，保持稳定姿态，并实现安全接触（双支撑腿同时触垫）。次要目标是节约燃料（即减少引擎使用次数），但绝不能以牺牲成功着陆和安全为代价。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心目标是到达并稳定在指定目标垫，属于典型的导航/目标到达任务；燃料效率和时间因素为辅助优化，不构成权重相当的冲突多目标。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 默认 float32（连续值，接触标志为 0.0 或 1.0）  
- 各维度含义：
  - obs[0]：`x_position`，相对目标垫的水平坐标（负/正代表左右），可用于奖励计算 `reward_usable: true`
  - obs[1]：`y_position`，相对垫高度的垂直坐标，可用于奖励计算 `reward_usable: true`
  - obs[2]：`x_velocity`，水平线速度，可用于奖励计算 `reward_usable: true`
  - obs[3]：`y_velocity`，垂直线速度，可用于奖励计算 `reward_usable: true`
  - obs[4]：`body_angle`，机体偏转角（以弧度计），可用于奖励计算 `reward_usable: true`
  - obs[5]：`angular_velocity`，角速度，可用于奖励计算 `reward_usable: true`
  - obs[6]：`left_support_contact`，左支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`
  - obs[7]：`right_support_contact`，右支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0：`no_engine`，不做任何推力  
  - 1：`left_orientation_engine`，点燃左姿态引擎（通常产生顺时针力矩和微小侧向力）  
  - 2：`main_engine`，点燃主引擎（通常产生向上的主要推力）  
  - 3：`right_orientation_engine`，点燃右姿态引擎（通常产生逆时针力矩和微小侧向力）  

说明：动作是离散的推进器选择，每个时间步只能选择一种操作。因此奖励中无法直接获得连续推力值，但可基于动作类型判断是否使用了引擎，从而惩罚燃料消耗。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `termination_conditions` 原文，共三种终止原因：
1. **crash_or_body_contact（碰撞或身体接触）**  
   - 很可能是与地面或障碍物发生不当碰撞，通常代表失败。
2. **horizontal_position_outside_viewport（水平位置超出视口）**  
   - 机体横向飞出边界，代表失败。
3. **body_not_awake_or_settled（机体未醒或已稳定）**  
   - 语义存疑：可能表示机体已进入休眠状态（着陆后稳定静止）或未能稳定（翻倒后停止）。由于原文将它与前两类并列且未注明成功/失败，该终止条件本身是 **模糊的**。在典型同类任务中，机体成功着陆后会进入“睡眠”（awake=False），因而该条件可能是 **成功终止**。但在正式设计 reward 时，不能直接依赖该条件为成功标志，必须结合接触、位置、速度等观测自行判断。

### 5.2 步骤返回信息
`step()` 返回 `(state, masked_reward, terminated, False, {})`，其中 `info` 为空字典。因此：
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因 info 为空，无可用信号）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

- 允许使用：
  - `obs`：当前观测（8维向量）
  - `action`：所选离散动作（0~3 整数）
  - `next_obs`：下一时刻观测
  - `info`：空字典，因此无可用字段
  - `training_progress`：仅当 prompt 明确允许时使用（本例未允许，应采用 0.0 或忽略）
- 禁止使用：
  - `original_reward`（官方奖励被掩码，不可使用、不可还原）
  - 任何未声明的 info 字段
  - 任何未声明的观测切片或全局变量

## 7. 可用于奖励函数的信号
基于 `obs`、`action` 和 `next_obs`，可提取以下信号：
- **位置信号**：`obs[0]`、`obs[1]`（或 `next_obs[0]`、`next_obs[1]`），表示相对目标垫的水平和垂直偏差。
- **速度信号**：`obs[2]`、`obs[3]`，水平与垂直线速度。
- **姿态信号**：`obs[4]`（偏转角）、`obs[5]`（角速度）。
- **接触信号**：`obs[6]`、`obs[7]`，左右支撑腿是否触地。
- **动作/引擎信号**：`action` 值，用于判断是否使用了引擎（0为无推力，1、2、3均消耗燃料）。
- **附加信号**：还可以计算 `next_obs` 与目标的距离变化、速度变化等差分信号。

所有这些信号均源自观测，可直接用于塑造奖励，无需依赖 info。

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不存在于 info 或 step 返回中，不能直接使用。
- **剩余燃料值**：环境中未提供，因此无法直接基于燃料余量设计奖励。
- **目标垫绝对坐标**：任务描述已暗示目标垫在画面中央，但观测是相对值，绝对坐标不可用（不影响任务）。
- 因 info 为空，任何类似 `info["success"]` 或 `info["termination_reason"]` 的信号均不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: lander_with_two_legs
  actuator_type: main_thruster_and_orientation_thrusters
  contact_structure: two_leg_contacts
primary_objectives:
  - 到达目标垫中心：x → 0, y → 0
  - 稳定着陆：双支撑腿触地，小速度，小倾斜角
secondary_objectives:
  - 节省燃料：最小化非零动作的使用次数
  - 快速到达：隐含在任务描述中，但不可牺牲安全性
main_failure_risks:
  - 与地面或障碍物猛烈碰撞（crash_or_body_contact）
  - 横向飞出可用区域（horizontal_position_outside_viewport）
  - 着陆后翻倒或未能垂直稳定，导致 body_not_awake_or_settled 但状态不合格
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: `goal_approach`**  
  purpose: 引导 agent 向目标垫中心靠近（x→0, y→0）  
  why_required: 是任务核心，没有它 agent 无法学会到达目标  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 单独使用可能导致高速冲撞目标，需配合减速职责

- **role_id: `soft_landing`**  
  purpose: 在接近目标时鼓励低速、小角度、双脚触地  
  why_required: 确保安全着陆，避免碰撞或弹开  
  usable_signals: [obs[2], obs[3], obs[4], obs[6], obs[7], next_obs[2], next_obs[3], next_obs[4], next_obs[6], next_obs[7]]  
  risks: 若在全局范围惩罚速度，可能抑制探索；应在接近目标区时增强权重

- **role_id: `successful_termination_bonus`**  
  purpose: 在满足成功着陆条件时给予巨大正向奖励  
 



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

