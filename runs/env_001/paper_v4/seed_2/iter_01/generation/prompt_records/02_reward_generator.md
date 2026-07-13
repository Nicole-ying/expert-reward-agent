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
该环境是一个 2D 飞行器着陆任务。飞行器从目标着陆平台上方区域出发，受到随机初始扰动力。核心目标是引导飞行器尽快接近并稳定地停在中央着陆平台上，并最大限度减少引擎使用。成功的着陆应满足：位置靠近平台中心、相对垂直速度很小、身体姿态平稳、且两个支撑部件都与平台接触。附属目标是降低燃料消耗（即减少引擎推力动作的使用）并缩短耗时。任务**不**是简单的平衡或存活，而是以到达指定位置为根本驱动。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: **high**
reason: 主目标明确为“到达并稳定在终点平台”，附属目标（节能、快速）不构成同等权重的冲突优化项，核心属于目标到达任务族。

动力学子类型进一步判断为 **goal_approach_and_soft_contact**，它精准刻画了“接近目标→减速→保持姿态→安全软接触”的着陆动力学过程。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: 默认 `float32`（其中 contact 标志为 0.0 / 1.0）
- 每一维含义及奖励可用性：

| 维度索引 | 名称 | 含义 | reward_usable |
|-----------|------|------|---------------|
| 0 | `x_position` | 飞行器相对于着陆平台的水平坐标 | true |
| 1 | `y_position` | 飞行器相对于平台高度的垂直坐标 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度 | true |
| 4 | `body_angle` | 机体倾斜角 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿是否接触地面（0/1） | true |
| 7 | `right_support_contact` | 右支撑腿是否接触地面（0/1） | true |

所有维度均可安全用于奖励函数计算，不存在需禁用的字段。

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- 动作含义：

| 动作索引 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 无任何推力输出 |
| 1 | `left_orientation_engine` | 开启左侧转向引擎（产生逆时针旋转力矩及少量推力） |
| 2 | `main_engine` | 开启主引擎（产生强大的向下推力，抵消重力） |
| 3 | `right_orientation_engine` | 开启右侧转向引擎（产生顺时针旋转力矩及少量推力） |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当飞行器身体不再活跃（进入休眠）且已基本静止时触发。通常由稳定着陆引发。
- **failure-like termination**: `crash_or_body_contact` —— 飞行器与地面/平台发生不当碰撞（如侧面触地、强烈冲击），视为坠毁；`horizontal_position_outside_viewport` —— 水平位置超出视觉窗口，表示偏离过大。
- **ambiguous termination**: 无。所有终止条件均可明确区分为成功或失败。
- **truncation**: 环境描述未提及最大步数截断，但实际部署时可能存在；truncation 不能被直接当作成功/失败信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （`info` 字典为空，无直接标志位）
- explicit_failure_flag_available: **false** （同上）
- allowed_info_fields: **无** 额外允许字段（`info = {}`）
- forbidden_or_uncertain_info_fields: 所有未在 step source 中出现的 `info` 键均不可用，包括但不限于 `success`、`failure`、`termination_reason`。

因此，奖励函数**必须**仅依赖 `obs`、`action`、`next_obs` 以及终止状态来隐式推断任务成果，不能依赖外部成功/失败标记。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：当前观测（8维数组）
- `action`：当前动作（int 或 one-hot，具体由环境决定；此处为 int）
- `next_obs`：下一时刻观测
- `info`：仅允许使用明确声明的字段（当前为空，故不可用）
- `training_progress`：当且仅当 prompt 明确允许时才能使用

**严格禁止使用：**
- `original_reward` / `official_reward`（已被屏蔽）
- 任何未在观察空间说明中出现的 `obs`/`next_obs` 切片
- 任何未在 `info` 中明确声明的字段

## 7. 可用于奖励函数的信号
- **位置信号**: `x_position`, `y_position` （相对目标，可直接计算距离）
- **速度信号**: `x_velocity`, `y_velocity` （衡量平缓着陆的关键）
- **姿态信号**: `body_angle` （水平姿态维稳）
- **角速度**: `angular_velocity`
- **接触信号**: `left_support_contact`, `right_support_contact` （软着陆的最后确认）
- **动作/引擎信号**: 动作索引可识别是否使用主引擎或转向引擎，用于燃料惩罚。
- **其他潜在信号**: 可从 `obs` 和 `next_obs` 差分得到加速度等，但需谨慎引入。

## 8. 不确定或不可用的信号
- 明确的目标位置坐标：由于观测是**相对于目标平台**的坐标，目标为 (0,0) 可自然推导。
- 明确的成功/失败标记：不存在。
- 剩余步数、燃料量等额外状态：不可用。
- 风力扰动等外部信息：被屏蔽，不可观测。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two support legs
  actuator_type: one main downward thruster + two lateral orientation thrusters
  contact_structure: two foot contacts on landing platform
primary_objectives:
  - 将飞行器运动到目标平台中心（x≈0, y≈0）
  - 在接触时保持极低的线速度和角速度（软着陆）
  - 着陆后两腿均触地且姿态基本水平
secondary_objectives:
  - 最小化燃料消耗（减少主引擎及转向引擎使用）
  - 尽可能快速完成着陆（隐式奖励通过效率导向）
main_failure_risks:
  - 坠毁或硬着陆（垂直速度过大导致 crash_or_body_contact）
  - 水平漂移出视口
  - 长时间盘旋消耗燃料无法完成着陆
  - 仅单腿着陆导致姿态不稳
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id: `position_proximity`**
  purpose: 鼓励飞行器向目标中心靠近。
  why_required: 到达目标是根本任务，无此职责 agent 不会学习定向移动。
  usable_signals: [`x_position`, `y_position`]
  risks: 单纯距离奖励可能造成高速撞击；需配合着陆职责。

- **role_id: `soft_landing_velocity`**
  purpose: 惩罚接近地面时的高线速度，尤其是垂直接触速度，促使轻缓落地。
  why_required: 防止以高速 crash 终止，是安全着陆的直接约束。
  usable_signals: [`y_position`（用于激活阈值）, `x_velocity`, `y_velocity`]
  risks: 若在高中空即施加强惩罚，会阻碍下降；需要结合高度门控。

- **role_id: `stable_orientation`**
  purpose: 抑制机体倾斜，要求着陆时姿态接近水平。
  why_required: 倾斜着陆极易单脚先触地导致 crash 或姿势失衡。
  usable_signals: [`body_angle`]
  risks: 角度惩罚过强可能使 agent 害怕使用转向引擎，影响水平位置控制。

- **role_id: `contact_completion`**
  purpose: 迫使者陆后两腿均与平台接触（仅接触地面但单腿不算完美）。
  why_required: 任务明确需要“安全接触”，双腿触地才是稳定终止条件的主要前兆。
  usable_signals: [`left_support_contact`, `right_support_contact`]（可与终止时或着陆后 next_obs 结合使用）
  risks: 提前给予接触奖赏可能导致虚假接触行为，必须与位置、速度、姿态强耦合。

- **role_id: `fuel_efficiency`**
  purpose: 惩罚使用引擎推力，降低燃料消耗。
  why_required: 明确要求“as little engine thrust as possible”，是任务描述的一部分。
  usable_signals: [`action`]（动作索引 1,2,3 对应不同引擎）
  risks: 惩罚过大会使 agent 选择无推理漂移，放弃着陆；惩罚太小则无法体现节能要求。

### 10.2 条件职责 conditional_roles
- 无额外条件职责。任务目标已由上述主职责完全覆盖。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: `speed_penalty_early`**
  reason: 在所有高度下均惩罚高速度可能严重阻碍下降进程，应改为仅在接近地面（y 较小）时激活。
  forbidden_or_missing_signals: 无（就是使用方式问题）

- **role_id: `force_fast_landing`（时间惩罚）**
  reason: 任务确有“as fast as possible”要求，但**缺失剩余时间或步数信号**，无法直接构建可靠的时间效率奖励，强行使用步数倒扣容易导致过早坠落或投机行为。建议避免。
  forbidden_or_missing_signals: 步数/剩余时间不可用

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `position_proximity` | `x_position`, `y_position` | 无 | `dense_state_signal` (negative distance / exponential decay) | 可直接用欧几里得距离的相反数或高斯式强度 |
| `soft_landing_velocity` | `y_position`, `y_velocity`, `x_velocity` | 无 | `bounded_signal` (高度门控 → quadratic_penalty) | 仅当 y 低于阈值时才激活速度惩罚项 |
| `stable_orientation` | `body_angle` | 无 | `quadratic_penalty` 或 `abs_penalty` | 需避免与转向引擎惩罚冲突 |
| `contact_completion` | `left_support_contact`, `right_support_contact` | 显式的“成功着陆”标志 | `binary_bonus` / `joint_contact_condition` | 宜在终止步或最后几步给予，且必须配合位置/速度检验 |
| `fuel_efficiency` | `action` (0,1,2,3) | 无 | `discrete_action_cost` | 可为主引擎动作分配更高惩罚，转向引擎较低 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 着陆速度过高导致 crash | 终止前几步 `y_velocity` 绝对值很大；终止于 crash | 增大 `soft_landing_velocity` 角色权重，或降低高度触发阈值 |
| 悬停漂浮不下降 | `y_position` 长时间保持较高，动作长期选择 0 或仅转向 | 调整 `position_proximity` 奖励陡度，强调 y 接近 0 的收益；或轻度放松燃料惩罚 |
| 单脚着陆不稳定 | 终止时仅一条腿 contact=1，另一条=0 | 强化 `contact_completion` 对双腿同时接触的要求；结合 `body_angle` 做耦合约束 |
| 水平漂移出视口 | 终止于 `horizontal_position_outside_viewport` | 增大 `position_proximity` 中 x 方向分量的惩罚，或加入 x 速度在远距离时的阻尼项 |
| 过度使用转向引擎导致旋转失控 | `angular_velocity` 持续很大，角度来回振荡 | 引入对 `angular_velocity` 的额外惩罚（与 `stable_orientation` 联合） |
| 为节省燃料完全不使用主引擎 | 任务长期无下降，终止于超时或 crash | 动态调整 `fuel_efficiency` 权重，或仅在接近地面时收紧引擎惩罚 |

此环境



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
