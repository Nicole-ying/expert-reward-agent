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

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
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
该环境是一个二维飞行器/着陆器轨迹优化任务。飞行器从视口顶部中心附近释放，并受到随机初始力的扰动。核心目标是**高效、安全地到达目标着陆垫，并在其上稳定停靠**。具体来说，智能体必须学会：
- 快速飞向目标垫（水平与垂直方向同时）
- 在接近时减速，避免猛烈撞击
- 维持稳定姿态，以正确角度与垫面接触
- 以两个支撑腿平稳触地并静止

次要目标是**尽量节省燃料**（减少发动机使用）。该任务不应被理解为单纯的燃料最小化问题，也不等同于保持平衡存活；到达目标才是主任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，即导航至目标并稳定停靠，这是典型的到达目标位置任务。虽然存在能耗和速度方面的附属要求，但它们不构成同等权重的多目标冲突，核心评价标准是能否成功着陆。

动力学上进一步细分为 **goal_approach_and_soft_contact**，因为任务不仅要求到达，还要求在接触垫面时姿态稳定、速度较小、且双脚同时或依次安全触地。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（推断，实际均为连续值或布尔值转换成的浮点数 0.0/1.0）
- 各维度含义（索引从0开始）：
  - obs[0]: x_position，水平相对目标垫中心的坐标，奖励可用：true（反映水平接近程度）
  - obs[1]: y_position，垂直相对垫面高度的坐标，奖励可用：true（反映高度及着陆进度）
  - obs[2]: x_velocity，水平线速度，奖励可用：true（用于抑制水平晃动或撞击）
  - obs[3]: y_velocity，竖直线速度，奖励可用：true（着陆瞬间的垂直速度关键）
  - obs[4]: body_angle，躯体朝向角度，奖励可用：true（姿态稳定是成功接触的前提）
  - obs[5]: angular_velocity，角速度，奖励可用：true（姿态晃动惩罚）
  - obs[6]: left_support_contact，左支撑腿接触标志（1.0接触，0.0未接触），奖励可用：true（可判定着陆状态）
  - obs[7]: right_support_contact，右支撑腿接触标志，奖励可用：true（同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 动作0：no_engine，不启动任何发动机（惯性滑行）
  - 动作1：left_orientation_engine，启动一个姿态控制发动机（推测产生逆时针或某一方向力矩，用于调整躯体角度）
  - 动作2：main_engine，启动主发动机（推测产生与躯体方向相反的推力，用于减速和上升）
  - 动作3：right_orientation_engine，启动另一个姿态控制发动机（与动作1反向的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：
  - `body_not_awake_or_settled` ：当身体不再“清醒”（即进入睡眠状态）或已稳定停靠时触发。这通常意味着飞行器已经在目标垫上静止，是成功的着陆迹象。
- **failure-like termination**：
  - `crash_or_body_contact` ：发生碰撞或主体直接接触（可能触地时姿态太差或超出容忍范围）
  - `horizontal_position_outside_viewport` ：水平位置移出视口，偏离着陆区域太远
- **ambiguous termination**：无
- **truncation**：未提供，但环境可能有额外的时间步限制，但 step 源中未体现，故不依赖。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （无明确的 info 字段标明成功）
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典 {}，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段都不可用，因为 step 返回空字典。

注意：终止条件`body_not_awake_or_settled`虽暗示成功，但它本身是终止标志，并非附加的成功/失败信号。若在奖励函数中依赖该状态，只能通过`done`旗标判断回合结束，而不能区分成功或失败。因此，奖励函数不能直接使用 `info["success"]` 等不存在字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
- **允许使用**：
  - `obs`：当前观测（8维）
  - `action`：当前动作
  - `next_obs`：下一观测（8维）
  - `info`：必须为空字典，不可使用任何键
  - `training_progress`：默认为0.0，仅当提示明确允许时才使用（本次任务未要求），暂不可用
- **禁止使用**：
  - `original_reward` 或任何官方奖励值
  - 未声明的 `info` 字段
  - 未声明的 `obs` 切片或隐含的内部状态

## 7. 可用于奖励函数的信号
- **位置信号**：
  - `x_position`（obs[0]）：可用于衡量水平接近程度
  - `y_position`（obs[1]）：可用于判定是否接近垫面
- **速度信号**：
  - `x_velocity`（obs[2]），`y_velocity`（obs[3]）：着陆瞬间或过程减速
- **姿态信号**：
  - `body_angle`（obs[4]）：姿态稳定与对准
  - `angular_velocity`（obs[5]）：姿态晃动抑制
- **接触信号**：
  - `left_support_contact`（obs[6]），`right_support_contact`（obs[7]）：确认腿部触地，可用于奖励稳定着陆
- **动作/发动机**：
  - 动作编号（0-3），可区分是否使用主发动机或姿态发动机；可用于节约燃料的惩罚

## 8. 不确定或不可用的信号
- 地面真实位置（如目标垫的确切坐标）：不可直接获得，但观测已经转换为相对值，故实际足够。
- 燃料消耗量：未提供，只能通过动作选择间接推断。
- 飞行器质量或惯量：不可用。
- 其他未公开的内部状态（如“awake”标志）：不可直接用于奖励。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_lander
  actuator_type: discrete_thrusters
  contact_structure: two_point_contact_landing_legs
primary_objectives:
  - 到达目标着陆点（最小化相对位置）
  - 在垫面上稳定停靠（双支撑腿接触且速度接近零）
secondary_objectives:
  - 尽量减少发动机使用次数（节能）
  - 尽快完成着陆（隐含时间效率）
main_failure_risks:
  - 水平漂移出视口
  - 姿态失控导致撞击或侧翻
  - 着陆速度过大造成“crash”
  - 过度使用主发动机飞得太高或再次远离目标
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_and_landing**
  - purpose: 引导飞行器向目标垫移动并最终着陆
  - why_required: 这是任务核心成功标准，必须通过距离减小、速度衰减、姿态对准来塑造。
  - usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  - risks: 过分强调距离惩罚可能让智能体高速冲向垫面，必须结合速度惩罚；接触信号不应过早使用，否则会鼓励提前触地而非飞至垫上。

- **role_id: soft_contact_stabilization**
  - purpose: 确保着陆时双支撑腿同时或先后安全触地，且速度、姿态均符合要求
  - why_required: 单腿着陆或翻滚都会导致 `crash_or_body_contact` 失败；稳定触地后需要保持静止以触发 `body_not_awake_or_settled`。
  - usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity]
  - risks: 当飞行器还在空中时，不应因为接触为0而惩罚，只能作为着陆后的正面奖励。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**
  - condition_to_use: 当智能体已经开始接近目标（例如距离已经小于某个阈值）或者可以保证任务成功时，加入动作惩罚以鼓励节省燃料。
  - usable_signals: [action]
  - risks: 过早施加动作惩罚会损害探索，导致智能体什么都不做或不敢使用发动机；必须渐进或仅在任务后期增强。

- **role_id: time_efficiency**（可选）
  - condition_to_use: 若需要鼓励较快到达，可通过给予每个时间步小额负奖励实现；但此环境未提供时间步信息，可以借助恒定的每步惩罚（如与action无关的生存惩罚）。需谨慎使用，避免智能体为快速而撞击垫面。
  - usable_signals: 无独立时间信号，只能以恒定的常数惩罚作为时间驱动力。
  - risks: 可能牺牲安全性。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: precise_posture_alignment**
  - reason: 任务只需要大概的姿态稳定（避免侧翻），不必追求极精确的0度角；过度要求角度可能限制智能体调整姿态的能力。
  - forbidden_or_missing_signals: 无精确姿态目标值，只需在着陆阶段大致垂直，可用但需放松。

- **role_id: exploration_bonus**
  - reason: 任务并非稀疏探索型，距离信号直接提供持续反馈，无需刻意增加好奇心或探索奖励。
  - forbidden_or_missing_signals: 无必要。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_and_landing | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | 无 | distance_penalty, velocity_penalty, angle_penalty, contact_bonus | 距离使用欧几里得或加权；速度在接近地面时加重惩罚；接触作为正面里程碑奖励 |
| soft_contact_stabilization | left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity | 无 | contact_reward (both legs), zero_velocity_bonus, angular_velocity_penalty | 当双腿接触且速度极低时给予高奖励，否则即使接触也极轻微奖励 |
| fuel_efficiency | action | 无 | action_penalty（除no_engine外），或main_engine_penalty | 对动作2（主发动机）可单独增加惩罚，因其耗能最大 |
| time_efficiency | 无专门信号 | time_step | constant_step_penalty | 需谨慎，防止过早终止 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器过度依赖主发动机，垂直悬浮或飞离目标 | y_position 持续增大或保持较大，而 x 偏离；中期奖励仍为正 | 增大距离惩罚随垂直高度的变化，防止悬停；对持续高耗动作加重惩罚 |
| 水平方向漂移出视口 | 终止于 `horizontal_position_outside_viewport`；训练中水平奖励未改善 | 加强水平位置惩罚，或在视口边缘施加额外斥力信号 |
| 着陆时姿态恶化/侧翻 | 终止于 `crash_or_body_contact`；观察 body_angle 和 angular_velocity 在接近地面时大幅波动 | 在接近垫面且速度较低时，增加姿态角惩罚的权重；对姿态发动机的使用给予宽容（前期不加过大惩罚） |
| 智能体只学习不用发动机，自由落体坠毁 | 低 episode 长度，无接触；奖励多来自无动作惩罚 | 延迟燃料效率奖励，确保任务成功率达标后再引入 |
| 过早单腿触地导致翻滚 | 只有一条腿接触时，身体角速度突然增加后 crash | 在接触奖励中严格要求双腿同时接触，或对单腿接触给予中性奖励而不停步加分 |



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


```
