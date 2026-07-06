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
控制一个平面单腿关节体持续向前跳跃，同时保持躯干直立且各关节处于健康范围内。摔倒、躯干倾角过大或任何身体状态超出安全范围将立即终止回合。策略的核心是产生持续的前向位移，而非仅仅保持平衡。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: (11,)
- dtype: 连续浮点数（具体精度由环境决定，通常为 float32）
- obs[0]: torso_height – 躯干（主体）的垂直高度
- obs[1]: torso_angle – 躯干的倾斜角度（方向角）
- obs[2]: upper_joint_angle – 上腿关节角度
- obs[3]: lower_joint_angle – 下腿关节角度
- obs[4]: foot_joint_angle – 足部关节角度
- obs[5]: forward_velocity – 躯干的水平前向速度
- obs[6]: vertical_velocity – 躯干的垂直速度
- obs[7]: torso_angular_velocity – 躯干的角速度
- obs[8]: upper_joint_speed – 上腿关节角速度
- obs[9]: lower_joint_speed – 下腿关节角速度
- obs[10]: foot_joint_speed – 足部关节角速度

## 4. 动作空间 action_space
- type: Box（连续）
- shape: (3,)
- 每个维度取值范围：[-1.0, 1.0]
- action 0: upper_joint_torque – 施加在上腿铰链的力矩
- action 1: lower_joint_torque – 施加在下腿铰链的力矩
- action 2: foot_joint_torque – 施加在足部铰链的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无明确定义的成功终止条件。任务目标是持续前进，没有预设的成功状态（如到达某个位置）。
- failure-like termination:
  - body_height_outside_healthy_range：躯干高度超出健康范围（典型为摔倒在地面或跳跃过高）
  - torso_angle_outside_healthy_range：躯干倾角过大（身体过度倾斜）
  - state_value_outside_finite_healthy_range：任何状态值变为无限/NaN（数值不稳定，通常由关节极限或碰撞导致）
- ambiguous termination: 无
- truncation: time_limit_reached（达到最大步数限制），为环境截断，不代表成功或失败，仅表示时间耗尽。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，包括 reward_forward, reward_ctrl, reward_survive, x_position, y_position, z_distance_from_origin 等；这些字段被明确禁止，且在实际环境中已被屏蔽。terminated/truncated 标志不通过 info 提供，也无法在 reward 函数中直接使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：形状 (11,) 的 numpy 数组，当前步的观察
- action：形状 (3,) 的 numpy 数组，当前步执行的动作
- next_obs：形状 (11,) 的 numpy 数组，执行动作后的观察
- info：固定为空 dict {}，不可从中读取任何字段
- training_progress：仅当 prompt 明确允许时才使用，否则忽略

禁止使用：
- original_reward：已屏蔽，不得依赖
- official_reward：等价于 original_reward，不可用
- info 中的所有字段：当前 info 为空，任何尝试读取都会失败
- 未在观察空间中声明的任何数据

## 7. 可用于奖励函数的信号
- position: 躯干高度（obs[0]），躯干倾角（obs[1]），各关节角度（obs[2‑4]）
- velocity: 前向速度（obs[5]），垂直速度（obs[6]），躯干角速度（obs[7]），关节角速度（obs[8‑10]）
- orientation: 躯干倾角可直接使用
- contact: 无直接接触传感器，但躯干高度过低可间接反映脚着地或摔倒
- action/engine: 动作力矩值（动作向量），可用于调节幅度或平滑性约束

## 8. 不确定或不可用的信号
- 真实的全局前向位移（x_position）被隐藏，不可用
- 任何形式的官方奖励分量（reward_forward, reward_ctrl, reward_survive 等）被屏蔽
- 健康状态的二元标志（身体是否超出范围）无法直接获得，只能从观察值推测
- 终止信号（terminated）不传入 reward 函数，不可用于奖励计算



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- locomotion_continuous_control：任务目标是稳定前进通过地形。重点观察 fast_then_fail / hover_or_stand_still / over_conservative_policy。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### forward_progress_reward
- 角色: 前进方向引导
- 数学形态: lambda_fwd * forward_velocity
- 需要信号: forward velocity component
- 使用说明: 奖励沿前进方向的速度。适合 locomotion 类任务。
- 风险: 快速前进但容易摔倒。
- 后续迭代: 若 fast_then_fail，配合稳定性约束。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### alive_bonus
- 角色: 存活激励
- 数学形态: lambda_alive * I[not_done]
- 需要信号: done flag
- 使用说明: 每步给予小额存活奖励，鼓励 agent 不提前终止。适合 locomotion/balance 类任务。
- 风险: hover_or_stand_still（原地不动来获取存活奖励）。
- 后续迭代: 若 agent 不动，减小权重或配合前向奖励。

### action_smoothness_penalty
- 角色: 动作平滑约束
- 数学形态: -lambda_smooth * |action - previous_action|
- 需要信号: previous action or action history
- 使用说明: 惩罚动作的剧烈变化。适合连续控制任务。
- 风险: 离散动作空间不可用（无历史信息）。
- 后续迭代: 若动作抖动，增大权重。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。
```
