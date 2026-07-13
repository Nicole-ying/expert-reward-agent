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
本环境是一个 2D 刚体飞行轨迹优化任务。  
**主目标**：从初始位置尽快到达画面中心的固定目标垫，并稳定停靠其上（速度接近于零，姿态保持平稳，左右支撑脚同时接触）。  
**次目标**：在保证成功到达的前提下，尽可能少地使用主引擎和姿态引擎推力。  
**不应混淆**：这**不是**纯粹的生存平衡任务，也不是多目标冲突的权衡——到达停靠是唯一的核心结果，燃料经济性只是“越快/越省”的优化维度，不构成独立的并行目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是“到达指定目标位置并稳定停靠”，存在明确的相对位置（观察空间前两维），终止条件包含“身体未唤醒或已稳定停靠”，动作空间控制推力与姿态，典型的目标导向导航形态。次目标（最少推力）可作为条件优化，但并不是与到达等效的平行目标，因此不选择 multi_objective_task。

动力学子类型：  
dynamics_subtype: goal_approach_and_soft_contact  
理由：飞行器需要趋近目标、减速、保持姿态、并最终以低速、稳定的身体接触停在垫上，完全匹配“接近目标并低速、稳定接触/停靠”的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（源信息未明确，但观测字段多为浮点数，接触标志以浮点 0.0/1.0 表示）
- 各维含义及奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫中心的水平坐标 | true |
| 1 | y_position | 相对于垫面高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 机体俯仰/倾斜角度 | true |
| 5 | angular_velocity | 机体角速度 | true |
| 6 | left_support_contact | 左支撑脚接触标志（1.0 接触，0.0 未接触） | true |
| 7 | right_support_contact | 右支撑脚接触标志 | true |

注意：所有 obs 均可在奖励函数中使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作 ID | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不执行任何推力，仅受物理影响 |
| 1 | left_orientation_engine | 点燃一个姿态引擎（产生转矩，改变角度） |
| 2 | main_engine | 点燃主引擎（产生沿当前朝向的推力） |
| 3 | right_orientation_engine | 点燃相反姿态引擎（与动作 1 反向的转矩） |

动作在奖励函数中可用，例如用于计算推力消耗的惩罚。

## 5. step 与终止条件分析
### 5.1 终止模式
源 step 中定义：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```

- **crash_or_body_contact**：发生碰撞或机身其他部位（非支撑脚）接触地面 → 可能为**失败**。
- **horizontal_position_outside_viewport**：水平位置偏离视口 → **明确失败**。
- **body_not_awake_or_settled**：身体进入休眠状态（失败）或已稳定停靠（成功）。该条件同时包含成功与失败两种情况，需根据状态推断 → **模棱两可的终止**。

此外未设置截断（truncation 恒为 False）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available: false**
  - info 字典为空，没有类似 `"success"` 的布尔标志。
- **explicit_failure_flag_available: false**
  - 虽然 crash 和出界显然为失败，但 step 源码未提供单独标志。
- **allowed_info_fields**: 空（info={}），奖励函数中**禁止**依赖任何 info 键。
- **forbidden_or_uncertain_info_fields**:
  - 任何未出现在源码中的 info 字段均禁止使用（如 `success`, `failure_reason` 等）。
  - 终止条件 `body_not_awake_or_settled` 未作为信号传入奖励，故无法直接区分，必须通过下一条终止时的 next_obs 特征（位置、速度、接触等）自行推测成功与否。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：上一步观察（所有维度）
- `action`：被执行的离散动作（0~3）
- `next_obs`：步骤后的观察（所有维度）
- 无其他可用 info 字段（info 为空）

禁止使用：
- `original_reward`：官方奖励被屏蔽，严禁依赖
- `info` 中任何未声明的键
- `training_progress`：当前提示未明确允许使用，故禁止（若后续明确可用则允许）

在不违反上述规则的前提下，可以利用 `next_obs` 在最后一步计算终止奖励（如对成功停靠给予大正奖励、对碰撞或出界给予负奖励）。

## 7. 可用于奖励函数的信号
以下归纳主要信号组：

- **位置相关**：`x_position`（水平偏差）、`y_position`（垂直高度）
- **速度相关**：`x_velocity`、`y_velocity`
- **姿态相关**：`body_angle`、`angular_velocity`
- **接触相关**：`left_support_contact`、`right_support_contact`
- **动作/引擎使用**：`action`（可解析为主引擎、姿态引擎是否点火）
- **其他**：可构造的组合信号，如到目标的欧氏距离、速度范数、角度误差、是否双腿同时接触（ `left_support_contact > 0.5 and right_support_contact > 0.5` ）作为停靠标志。

## 8. 不确定或不可用的信号
- **成功/失败标志**：不存在显式 flag，需根据终止条件推定
- **目标垫半径或停靠判定阈值**：环境描述未给出精确的几何尺寸与速度阈值，需通过观察大量轨迹或保守假设（如距离 < 0.1，速度 < 0.05，角度 < 0.1，双腿接触）来推断成功
- **燃料量或推力大小**：环境中推力大小被屏蔽，只能以二元动作是否发生作为粗略指标（消耗主引擎/姿态引擎的次数）
- **阶段信息（进度）**：无 progress 字段，training_progress 当前禁止使用
- **环境原始奖励**：已屏蔽，不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 刚体飞行器
  actuator_type: 一个主推力引擎、两个差动姿态引擎
  contact_structure: 左右两个支撑脚，接触垫面后支撑机身，要求平稳双足触地
primary_objectives:
  - 水平与垂直位置收敛至目标垫区域（距离极小）
  - 速度衰减至近乎为零（软着陆）
  - 姿态角稳定至竖直附近（防止倾覆）
secondary_objectives:
  - 最小化主引擎与姿态引擎的使用次数（降低推力消耗）
  - 尽快完成停靠（隐含在速度与距离衰减中）
main_failure_risks:
  - 过早与地面碰撞或机身其他部分触地（crash）
  - 水平漂移出界
  - 速度过大或单侧脚先着地导致翻倒/未稳定
  - 姿态角发散，无法控制翻转
  - 为了省力而过度悬停，长时间无法停靠
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_target**  
  purpose: 驱使飞行器向目标垫靠近，缩小水平与垂直偏差。  
  why_required: 到达目标垫是任务的核心结果。  
  usable_signals: `x_position`, `y_position`（或组合成距离）  
  risks: 惩罚过重可能导致急加速撞地；需与速度控制协调。

- **role_id: soft_landing**  
  purpose: 在靠近目标时强制减速，避免高速冲击导致 crash 或不稳定接触。  
  why_required: 停靠不仅要求位置正确，还要求速度极低。  
  usable_signals: `x_velocity`, `y_velocity`（速度范数）  
  risks: 过早强制减速可能延长训练时间；需结合距离权重，仅在接近目标时增强。

- **role_id: stable_upright**  
  purpose: 保持机体姿态接近竖直，防止因倾斜导致单脚着地或翻倒。  
  why_required: 稳定停靠需要双足同时接触，姿态偏差过大会破坏这一条件。  
  usable_signals: `body_angle`, `angular_velocity`（以及双脚接触标志作为验证）  
  risks: 过度强调可能使动作选择保守；可仅当接近垫面时提升权重。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**  
  purpose: 最小化推力使用，避免不必要的引擎点火。  
  condition_to_use: 当飞行器已经学会基本到达和减速后，或者通过训练进度动态调控；初期训练可关闭或降低权重，防止学习停滞。  
  usable_signals: `action`（解析为主引擎、姿态引擎是否点火）  
  risks: 过早引入会与到达、减速目标冲突，导致 agent 选择 no_engine 而悬停不前。

- **role_id: terminal_success_bonus**  
  purpose: 在 episode 结束时，若能识别成功停靠，给予一次性大幅奖励。  
  condition_to_use: 必须依据 `next_obs` 推断是成功停靠而非 crash/出界；简单判定条件：`abs(x_position) < 阈值, y_position < 阈值, sqrt(x_vel^2+y_vel^2) < 阈值, abs(angle) < 阈值, 左右接触 > 0.5`。  
  usable_signals: `next_obs` 的上述组合  
  risks: 阈值必须保守，否则可能误判 crash 为成功；应同时配合失败惩罚。

- **role_id: terminal_failure_penalty**  
  purpose: 对所有明显失败（水平出界、速度过大撞击）给予负奖励，防止 agent 落入不安全区域。  
  condition_to_use: 从 `next_obs` 能够确认的危险状态（如 `abs(x_position)` 非常大、`y_position` 突变、速度极大且无接触等）。由于无显式 crash 标志，需谨慎设计启发式条件。  
  usable_signals: `next_obs` 的位置、速度、接触  
  risks: 误将成功案例惩罚，破坏学习。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: early_contact_penalty**  
  reason: 本任务中双腿同时接触是停靠成功的一部分，对过早的单腿接触直接惩罚可能阻止 agent 学习降落；应在接触时检查是否满足成功条件。  
  forbidden_or_missing_signals: 缺少真正的“非稳定接触”标志，无法区分 landing 过程中的短时单脚触地和碰撞。

- **role_id: angular_smoothness**  
  reason: 现阶段任务关注终点姿态误差，而非过程角速度平滑。角速度惩罚与姿态稳定职责部分重叠，过度施加可能限制必要的姿态调整。  
  forbidden_or_missing_signals: 无直接的不良振荡度量。

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | `x_position`, `y_position` (obs/next_obs) | 无 | dense_state_signal (负距离), bounded_signal | 可用 `-sqrt(x^2+y^2)` 或分段线性 |
| soft_landing | `x_velocity`, `y_velocity` (obs/next_obs) | 无 | dense_state_signal (负速度范数), quadratic_penalty | 可结合与目标的距离作为乘数 |
| stable_upright | `body_angle`, `angular_velocity` | 无 | bounded_signal (负绝对值角度), damping_penalty | 角度误差小于阈值时不处罚 |
| fuel_efficiency | `action` (离散) | 推力大小 | per_action_binary_penalty | 对 action 1,2,3 施加小负奖励 |
| terminal_success_bonus | `next_obs` (位置、速度、角度、接触) | explicit_success_flag | large_discrete_bonus_upon_condition | 判断条件见 10.2 |
| terminal_failure



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

- target_score: 200.000
- best_score_so_far: -10.160

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| contact_reward + gated_goal_reward | 1 | -10.160 | -10.160 | unsolved |
| contact_reward + engine_penalty + goal_reward + height_reward + landing_bonus | 1 | -20.700 | -20.700 | unsolved |
| contact_bonus + engine_penalty + gated_goal + height_reward | 1 | -63.810 | -63.810 | unsolved |
| global_speed_penalty + progress_reward + soft_landing | 1 | -104.190 | -104.190 | unsolved |
| engine_penalty + landing_reward + lateral_penalty + progress_reward | 2 | -122.030 | -122.030 | unsolved |
| contact_reward + descent_reward + engine_penalty + landing_bonus | 1 | -123.190 | -123.190 | unsolved |

## Previous interventions

- iter 2 (score=-63.810, structure=contact_bonus + engine_penalty + gated_goal + height_reward): 修改方案：: 1. **强化接触奖励**：增加双腿接触的奖励权重，并额外对双腿同时接触给予大幅奖励，确保Agent有动力接触垫。
- iter 4 (score=-123.190, structure=contact_reward + descent_reward + engine_penalty + landing_bonus): 修改方案：: 用线性下降奖励（最高限制）替代倒数高度奖励，驱动agent持续下降。
- iter 6 (score=-122.770, structure=engine_penalty + landing_reward + lateral_penalty + progress_reward): 修改方案: 主奖励：`progress_reward` = `w_progress * (dist_now - dist_next)`，乘以安全门控因子。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
