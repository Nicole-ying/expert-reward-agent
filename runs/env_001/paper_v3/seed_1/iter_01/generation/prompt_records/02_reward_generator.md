# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个 2D 飞行器从顶部中央区域安全、快速地到达并稳定停靠在中央目标垫上。  
次目标：尽可能节省引擎推力。  
不应混淆的目标：纯粹的飞行姿态控制不是最终目标；省燃料不能影响成功着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是到达指定目标位置（中心垫），并稳定着陆；附属有速度、姿态、燃料等要求，但主次分明。

动力学子类型 dynamics_subtype: goal_approach_and_soft_contact  
原因：接近目标的同时需要低速、稳定接触，由两条支撑腿的接触标志和身体角度稳定要求共同体现。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (推测，其中的接触标志为 0.0 或 1.0)  
- 字段含义与可奖励性：

| 索引 | 名称 | 含义 | 可用于奖励？ |
|------|------|------|--------------|
| 0 | x_position | 相对目标垫的水平坐标 | 是 |
| 1 | y_position | 相对目标垫高度的垂直坐标 | 是 |
| 2 | x_velocity | 水平线速度 | 是 |
| 3 | y_velocity | 垂直线速度 | 是 |
| 4 | body_angle | 机体角度 (orientation) | 是 |
| 5 | angular_velocity | 角速度 | 是 |
| 6 | left_support_contact | 左支撑腿接触标志 (0/1) | 是 |
| 7 | right_support_contact | 右支撑腿接触标志 (0/1) | 是 |

所有 8 个量均可在奖励函数中安全使用；全部源自环境本身，无隐藏信息。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：

| 动作 ID | 名称 | 含义 |
|---------|------|------|
| 0 | no_engine | 无任何引擎推力 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生旋转力矩） |
| 2 | main_engine | 启动主引擎（产生主推力，可能同时影响姿态） |
| 3 | right_orientation_engine | 启动右姿态引擎（与左引擎方向相反） |

## 5. step 与终止条件分析
### 5.1 终止模式
- 可能表示成功的终止模式：
  - `body_not_awake_or_settled`：如果同时满足位置接近目标、速度极小、双支撑腿着地且角度稳定，可视为成功着陆。
- 可能表示失败的终止模式：
  - `crash_or_body_contact`：机体非腿部分与障碍物/地面碰撞。
  - `horizontal_position_outside_viewport`：超出水平边界。
- 模糊终止模式：
  - `body_not_awake_or_settled`：在偏离目标或仅有单腿接触时也可能触发，此时可能是失败（例如跌落后停止）。
- 截断：未提及任何时间截断或步骤限制。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] (info 固定为空字典)
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不得使用。终止原因不可从 info 获得。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs, action, next_obs（所有维度均可读）
- info：仅可使用明确声明的字段，当前为空，故不得依赖 info
- training_progress：本环境提示未提及允许使用，禁止使用

禁止使用：
- original_reward
- 任何未在观察空间中声明的信号或 info 字段
- 终止标志 (terminated) 不能直接传入奖励函数，不可用于奖励计算

## 7. 可用于奖励函数的信号
- **位置**：`x_position`, `y_position`（相对目标垫，可直接做距离度量）
- **速度**：`x_velocity`, `y_velocity`（控制着陆时的冲击力与稳定性）
- **姿态**：`body_angle`, `angular_velocity`（稳定竖直）
- **接触**：`left_support_contact`, `right_support_contact`（两条支撑腿是否接触）
- **动作/引擎**：`action`（离散动作索引，可用于燃料消耗度量）
- **其他**：无

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不可用（info 为空）
- 终止原因：不可用
- 碰撞类型或碰撞强度：不可用
- 剩余燃料或能量预算：不可用（只能由动作序列间接推断）
- 目标垫的具体半径或容差：未给出，需从位置坐标经验推测

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle/lander
  actuator_type: discrete (one main engine, two orientation engines)
  contact_structure: two support legs (binary contacts)
primary_objectives:
  - 到达并停留在目标垫上（位置接近、双支撑腿接触）
  - 低速、稳定着陆（速度小、角度竖直）
secondary_objectives:
  - 尽量少使用引擎推力（节省燃料）
  - 尽快完成着陆（非强制时间最小化）
main_failure_risks:
  - 机体非腿部分与地面碰撞
  - 水平漂出视口
  - 角度过大导致不稳定着陆或接触失败
  - 过早切断引擎导致粗猛着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity_reward
  purpose: 驱动机体向目标垫移动并停留在中心
  why_required: 任务核心是到达目标位置，没有此职责 agent 将无方向
  usable_signals: [x_position, y_position]
  risks: 单纯距离奖励可能引导高速冲撞，需配合速度控制

- role_id: soft_landing_velocity_penalty
  purpose: 在接近目标时抑制过大的水平与垂直速度
  why_required: 成功要求“settle”，高速冲击会导致失败或无法稳定
  usable_signals: [x_velocity, y_velocity, x_position, y_position]
  risks: 如果对所有时间惩罚速度，可能阻止探索；应只在靠近目标时激活

- role_id: orientation_stability_reward
  purpose: 保持机体角度稳定（接近 0 或竖直）
  why_required: 着陆需要平稳接触，大角度倾斜可能只接触单腿甚至倾覆
  usable_signals: [body_angle, angular_velocity]
  risks: 角度控制权重过大可能抑制必要的姿态调整；可通过绝对角度惩罚实现

- role_id: contact_reward
  purpose: 奖励双支撑腿同时接触
  why_required: 着陆成功的明确信号；能帮助 agent 理解“settle”状态
  usable_signals: [left_support_contact, right_support_contact]
  risks: 若只奖励接触而不要求位置，agent 可能学会到处触碰；必须与目标位置结合

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty
  purpose: 减少不必要的引擎使用
  condition_to_use: 当 agent 已经能稳定到达目标附近时，可以逐步引入动作惩罚；或在任意时刻加入极小权重，但确保不压倒主目标
  usable_signals: [action]
  risks: 过早或过大惩罚会使 agent 不敢移动，完全不做探索

- role_id: settlement_bonus
  purpose: 在完全满足着陆条件时给予一次性高奖励
  condition_to_use: 仅在极靠近目标、速度极小、双支撑腿接触且角度稳定的状态下给予；相当于“成功”奖励的代理
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 如果阈值过严可能过于稀疏，难以学习；过松则误触

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_crash_penalty
  reason: 缺少可靠的失败/碰撞标志，终止原因不可从 info 获得；若强行用 next_obs 推断崩溃（例如检测到全零）可能不可靠且引入噪声
  forbidden_or_missing_signals: [collision_flag, termination_reason]

- role_id: time_optimality_penalty
  reason: 任务描述要求“尽快”，但未提供时间步数度量或截断信息；主动针对步数设计奖励可能干扰主要目标，可作为次要考虑，不宜作为强约束
  forbidden_or_missing_signals: [step_count, truncation_flag]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity_reward | x_position, y_position | — | dense_state_signal, euclidean_distance, gaussian_mixture, potential_based | 可用 -distance^2 或 exp(-dist) 等；可配合 shaping |
| soft_landing_velocity_penalty | x_velocity, y_velocity, x_position, y_position | — | quadratic_penalty, masked_by_proximity | 仅在靠近目标时生效，设为平方惩罚 |
| orientation_stability_reward | body_angle, angular_velocity | — | quadratic_penalty, absolute_penalty | 可直接使用 -angle^2, -ang_vel^2 或线性惩罚 |
| contact_reward | left_support_contact, right_support_contact | — | boolean_and, discrete_bonus | 当两个接触标志均为 1 时给予正奖励；与位置耦合 |
| fuel_efficiency_penalty | action | fuel_consumption_per_action (隐含) | action_cost_lookup, scaling | 离散动作：非零动作给予小额惩罚；逐渐引入 |
| settlement_bonus | 全部位置/速度/角度/接触信号 | — | thresholded_bonus | 所有条件逻辑与，成功后给较大常数 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 离目标很远就停止（过早认为任务完成） | 终止时的 x_position, y_position 大，或单腿/无接触 | 加强 goal_proximity_reward，或降低 settlement_bonus 阈值 |
| 高速冲撞目标垫（crash 终止） | 终止前速度大，x,y 速度未衰减，或仅有极短时接触 | 加强 soft_landing_velocity_penalty 或在较远距离就施加减速 |
| 持续悬停不下降（不敢靠近或燃料恐惧） | y_position 缓慢变化，长期不终止，动作多为 0 | 减弱或延迟 fuel_efficiency_penalty，增加目标垫附近的吸引 |
| 角度剧烈摇晃无法稳定 | body_angle 和 angular_velocity 振幅大，接触断续 | 增加 orientation_stability_reward 权重，考虑角度变化率惩罚 |
| 单腿着地反复弹跳 | 仅一侧接触标志为 1，且 x,y 位置略偏 | 提升接触奖励，并对单腿接触给予轻微负奖励（需安全引入） |
| 燃料使用过度，但成功着陆 | 动作序列中大量非零动作 | 在训练后期逐步引入 fuel_efficiency_penalty，进行策略微调 |
| 水平漂出视口 | x_position 绝对值过大触发终止 | 早期强化位置惩罚，或用较大的位置奖励梯度防止漂移 |



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


```
