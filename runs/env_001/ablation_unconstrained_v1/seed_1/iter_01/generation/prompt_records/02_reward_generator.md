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
这是一个 2D 类飞行器着陆/姿态控制任务。  
主体从屏幕上方中部某位置出发，受到初始随机作用力。  
目标是**尽快到达并稳定降落在中央目标平台**上，同时**尽可能少使用引擎推力**。  
智能体需要学习：  
- 向目标平台靠近  
- 降低速度  
- 保持稳定姿态  
- 与平台安全接触（两条支撑腿同时着地）  

附属优化包括省燃料、时间短、姿态平稳，但不额外构成独立目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32 或 float64（具体未知，但连续数值）
- obs[0]: x_position —— 水平坐标，相对于目标平台（零点为目标中心）
- obs[1]: y_position —— 垂直坐标，相对于平台高度（零点为平台表面高度）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角度（可能用弧度）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志，1.0 表示接触，0.0 表示不接触
- obs[7]: right_support_contact —— 右支撑腿接触标志，1.0 表示接触，0.0 表示不接触

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine —— 不启动任何引擎，无推力/力矩
- action 1: left_orientation_engine —— 启动左姿态引擎（产生顺时针或逆时针力矩，具体方向需经验判断）
- action 2: main_engine —— 启动主引擎（产生向上的主要推力）
- action 3: right_orientation_engine —— 启动右姿态引擎（产生与动作1相反的力矩）

动作空间是针对飞行器的简单推力与姿态控制，无油门大小，每个动作固定输出一个脉冲。

## 5. step 与终止条件分析
### 5.1 终止模式
环境在满足以下任一条件时终止（terminated = True）：
- **crash_or_body_contact** —— 主体部分（非支撑腿）发生碰撞或接触（可能摔到地面上、撞到障碍等）
- **horizontal_position_outside_viewport** —— 水平坐标超出视口范围（远离目标平台过远）
- **body_not_awake_or_settled** —— 身体不再运动或已稳定（包括成功着陆后稳定静止）

> 可见，最后一种可能既包含成功着陆稳定，也可能包含其他静止的状态（如卡住、惯性停止等）。需要结合上下文判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false   （info 中没有 success 字段）
- explicit_failure_flag_available: false   （info 中没有 failure 字段）
- allowed_info_fields: （info 为空字典，无任何字段可用）
- forbidden_or_uncertain_info_fields: info 中所有字段不可用（因为 info 为空）；不可使用 terminated 标志

**注意**：终止条件（crash/出界/稳定）的真实语义不能直接从 `info` 获得，需要从 `next_obs` 的状态推理。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 转换前的当前观测
- action —— 刚刚执行的动作（整数0~3）
- next_obs —— 转换后的观测（用于推断状态变化和终止原因）
- info 中明确允许的字段 —— 当前 info 为空，所有 info 字段都禁止使用
- training_progress —— **当前任务描述未授权使用，默认禁止使用**，除非 prompt 明确说明允许

禁止使用：
- original_reward 或任何官方奖励
- 未声明的 info 字段
- 未声明的 obs 切片（例如不能假设 obs[0] 存在之外的含义）
- 任何形式的终止标志（terminated, done）

奖励函数设计时，**不得直接访问环境内部的终止条件**，只能通过 `next_obs` 的状态特征间接推断成功或失败。

## 7. 可用于奖励函数的信号
- **位置（接近目标）**：`next_obs[0]`（相对目标平台的水平距离），`next_obs[1]`（高度偏移），可鼓励 x,y 趋向 0
- **速度（着陆柔和性）**：`next_obs[2]`, `next_obs[3]`，可惩罚过大的速度，尤其垂向着陆速度
- **姿态**：`next_obs[4]`（身体角度），鼓励接近 0（水平姿态）；`next_obs[5]`（角速度），惩罚剧烈旋转
- **接触**：`next_obs[6]` 和 `next_obs[7]`（左右支撑腿接触），两条腿同时接触标志稳定的成功着陆
- **动作/引擎使用**：从 `action` 可以判断是否使用了引擎（0为无引擎，其他为有引擎），可惩罚燃料消耗；也可以结合燃料效率同时使用 `obs` 与 `next_obs` 的变化判断推力作用

## 8. 不确定或不可用的信号
- **成功/失败标签**：不存在于 info，不能直接获取
- **终止原因分类**：无法得知是 crash、出界还是稳定，只能从 next_obs 中的位置、速度、接触状态进行概率性推断
- **距离目标的确收指标**：没有明确的“着陆成功”布尔信号，需要通过位置足够近、速度接近零、双腿接触等组合条件来近似
- **训练进度信息**：默认不允许，除非将来 prompt 明确授权
- **任何隐藏的内部状态**：例如风力、摩擦力、随机初始冲量等，都不在观测中

---

**总结**：这个环境是一个典型的导航/着陆任务（navigation_goal_reaching），奖励需要从位置、速度、姿态、接触状态和动作中手工构造，以引导快速且省燃料地平稳着陆到平台中心。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- navigation_goal_reaching：任务目标是接近/到达指定位置。重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 使用说明: 多条件组合的完成近似。不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 使用说明: 当环境提供显式 success flag 时可用。若 explicit_success_flag_available=false，不可使用。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 使用说明: 惩罚每步耗时。先完成任务再加入，不建议 v1 使用。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 使用说明: 按条件切换奖励模式。v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。
```
