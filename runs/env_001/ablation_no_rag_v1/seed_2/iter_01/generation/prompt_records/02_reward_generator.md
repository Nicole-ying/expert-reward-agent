# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境背景；
2. expert_reward_context.md：RAG 检索并压缩后的专家知识；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务：
直接生成第一版奖励函数 `reward_v1.py`，并附带一份简短设计说明。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 route 推荐公式。
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

以下原则适用于所有环境和任务类型。expert_reward_context.md 中的骨架是奖励设计原语、风险提示和参考起点，不构成封闭候选集合。你可以直接采用、组合、变形这些原语，也可以根据任务目标和可用环境信号提出新的数学结构。

## 原则 1：信号可用性优先

- 先检查 environment_card.md 和 expert_reward_context.md 中声明的可用信号（obs indices、info 字段、termination 信息）。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号（如只在特定区域触发的 bonus）触发率过低时等于摆设。
- 连续函数（线性、bounded、乘积）比离散条件更有利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应在数值上无条件压制任务驱动力；具体尺度必须结合其触发频率、数学形态和预期行为判断，不能套用固定比例阈值。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同的时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号（如 distance 和 progress_delta），它们会产生重复梯度。
- 不要让惩罚项压制探索——过于严厉的姿态/速度约束可能导致 agent 不敢行动。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价（energy_penalty、time_penalty），agent 应先学会完成任务再优化效率。
- 复杂门控（gated_reward）、动态课程（dynamic curriculum）默认后续迭代再加入。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。例如：只奖励接近目标可能导致在目标附近震荡而不完成；只奖励接触可能诱导 agent 反复轻触地面。
- 如果使用 contact 或类似离散事件，必须和其他连续条件组合，不能单独作为成功信号。

## 原则 7：最小设计

- reward_v1 推荐使用 2~4 个组件，每个必须有明确角色，不能为了显得完整而堆叠。
- 信号多的时候要主动取舍——不是所有可用信号都需要放进 v1。优先选择对任务目标最核心的信号。
- 把 expert_reward_context.md 中按任务路由推荐的骨架作为思考面，而不是允许列表。最终设计由任务目标、可用信号和预期策略行为决定，不要求组件必须对应某个 skeleton_id。
- components dict 里只放 total_reward 公式中等号右边的变量。如 `total_reward = A + B + C`，则 components 只有 A、B、C。不要把 `total_reward` 放进 components，也不要把 A 的计算中间变量（如 A = a1+a2+a3 中的 a1）放进 components。

# role-based component budget

以上原则的具体落地方式：v1 推荐使用 2~4 个组件，按以下角色组织。检索骨架只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent "做什么能得分"。主信号的特征：
- 每步都有梯度（稠密），不是只在终点或特定事件触发
- 与任务目标直接相关（靠近目标、前进、保持存活等）
- 在策略学习中承担主要任务驱动作用；它不一定是每一步数值最大的正向项
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`

主信号可以来自检索骨架，也可以是对多个设计原语的组合、变形或基于环境事实构造的新形式。不要为了匹配骨架名称而牺牲任务语义。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/平滑约束。** 如果任务需要控制速度、姿态、动作抖动等，可以加入轻量惩罚项。约束的角色是"方向盘"而非"刹车"——权重应明显小于主信号。如果环境不需要（如离散动作空间不需要 action_smoothness），不加。
- **0~1 个任务完成近似信号（proxy）。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合（不是单一二值条件），且不能直接伪造 success flag。
- **0~1 个效率/动作代价。** 权重必须很小。v1 阶段 agent 应先学会完成任务，效率优化后续迭代再加入。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确的 termination_reason）
- gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）

## 避免重复

- 不要同时大权重使用两个计算同一物理量的信号（如 distance_reward 和 progress_delta_reward）。如果两个都用了，一个必须是主信号，另一个只能是小权重辅助锚点——但更建议只选一个。

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

为了兼容旧 wrapper，也可以把 components 写入 `info["reward_terms"]`，但仍然建议返回 `(float(total_reward), components)`。

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
- components 只包含被加到 total_reward 的组件（A、B、C），**不包含 total_reward 本身**。

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
- 使用了哪些奖励组件；
- 每个组件的角色；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些组件留到后续迭代；
- 训练后应该观察哪些 failure mode。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部中央附近出发，尽快飞到画面中心的着陆垫上并稳定停靠。过程中需要尽量节省引擎推力，同时避免发生碰撞、水平出界或提前休眠。最终期望的位置是着陆垫上方、接近静止、姿态稳定且左右支撑脚均与垫子接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据常规推断，其中 contact 标志用 0.0/1.0 表示）
- obs[0] (x_position): 飞行器相对着陆垫中心的水平坐标（可能是负、零或正）
- obs[1] (y_position): 飞行器相对着陆垫高度的垂直坐标（正值在上方，负值在下方）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（弧度或度？具体范围由物理实现决定，但在奖励中应处理为接近 0 表示竖直姿态）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）
- obs[7] (right_support_contact): 右支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine —— 什么都不做（不点火，惯性和风力影响下自由飘动）
- action 1: left_orientation_engine —— 点火某个方向的姿态控制引擎（用于向左旋转机体）
- action 2: main_engine —— 点火主引擎（提供向上的推力，用于减速或上升）
- action 3: right_orientation_engine —— 点火相反方向的姿态控制引擎（用于向右旋转机体）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 飞行器在着陆垫上方稳定着陆，速度很小，角度接近 0，且左右支撑脚均接触（即 body_not_awake_or_settled 触发，同时位置、速度、角度、接触均满足期望状态）。该情形隐含“成功到达目标并安全停靠”。
- failure-like termination: 
  - crash_or_body_contact（飞行器主体与地面或边界发生破坏性碰撞）
  - horizontal_position_outside_viewport（飞行器水平飞出画面）
- ambiguous termination: body_not_awake_or_settled 可能既包含成功着陆（满足条件），也可能包含飞行器在非目标位置提前休眠（例如坠落在远处后不动、或能量耗尽停止）。因此该终止条件本身不能直接等同于成功，必须结合位置、接触等信息判断。
- truncation: 源码中为 `False`，未显示任何截断机制，可假设无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无任何显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空，无有效字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，不能假设存在 success、failure 或其他标记）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- obs：上一时刻的完整 8 维观察（可在函数内部使用，但通常奖励基于下一状态 next_obs 更多）
- action：刚刚执行的动作（0~3）
- next_obs：执行动作后的完整 8 维观察，是奖励计算的主要依据
- info：当前 step 返回的 info，这里为空字典 `{}`，因此不能依赖其中的任何字段
- training_progress：未在 prompt 中声明需要使用，仅作为保留参数，默认不允许用于奖励计算，除非 prompt 明确允许

**禁止使用：**
- original_reward（已明令禁止）
- 任何未被允许的 info 字段（info 为空，无字段可用）
- 任何未在观察空间中声明的变量（如官方隐藏的内部状态）

## 7. 可用于奖励函数的信号
以下信号均可从 next_obs（或结合 obs 计算增量）中提取，且属于任务相关的可靠特征：

- position: x_position, y_position（obs[0], obs[1]）—— 可衡量距着陆目标的远近、是否处于垫子正上方
- velocity: x_velocity, y_velocity（obs[2], obs[3]）—— 速度大小影响安全着陆和平顺性
- orientation: body_angle（obs[4]）—— 姿态是否竖直（接近 0）
- angular_velocity: obs[5] —— 旋转是否稳定
- contact: left_support_contact, right_support_contact（obs[6], obs[7]）—— 是否双脚接触，判断着陆状态
- action/engine: 可依据动作是否使用主引擎或姿态引擎来鼓励节能（如惩罚主引擎使用）

## 8. 不确定或不可用的信号
- 没有任何来自 info 的成功/失败/终止原因等信号
- 无从判断系统内部是否已判定“任务成功”（只能通过 next_obs 的数值组合自行推断）
- 可能存在的引擎燃料余量、风力大小等内部物理量完全未暴露，不可使用
- 原始奖励 original_reward 被明确禁止，不可使用



# expert_reward_context.md

# Expert Reward Context Disabled

This run is the w/o Expert RAG ablation. Design from environment facts only.

```
