# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视野顶部中心出发，通过离散推力指令，以**尽可能短的时间**安全到达并稳定停在中央目标垫上，同时**尽可能少地消耗发动机燃料**。  
需要学会：快速靠近目标、降低速度、保持姿态稳定并实现安全接触。任务同时优化 **移动速度** 与 **燃料经济性**，属于典型的多目标权衡问题。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务描述明确要求同时优化“到达目标的速度”（时间）和“燃料消耗”（最少推力），是典型的双目标权衡问题，不属于单纯的导航、运动控制或抓取，故归类为多目标任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32（推测，连续量通常为 float32；contact 信号为 0.0 或 1.0）
- obs[0] (x_position): 飞行器水平坐标相对于目标垫中心的偏移量
- obs[1] (y_position): 飞行器垂直坐标相对于目标垫高度的偏移量
- obs[2] (x_velocity): 水平速度
- obs[3] (y_velocity): 垂直速度
- obs[4] (body_angle): 飞行器当前朝向角
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑脚/传感器是否与目标垫接触（1.0=接触, 0.0=未接触）
- obs[7] (right_support_contact): 右侧支撑脚/传感器是否与目标垫接触（1.0=接触, 0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete, n=4
- action 0: 无引擎（不产生任何推力）
- action 1: 左姿态引擎（点火左侧姿态调整发动机，产生绕质心的力矩或侧向推力）
- action 2: 主引擎（点火主推力发动机，通常产生主要向上/向后的推力）
- action 3: 右姿态引擎（点火右侧姿态调整发动机，作用与左姿态引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  飞行器到达目标垫附近，速度极低、姿态稳定、双支撑脚均接触（obs[6] 和 obs[7] 均为 1.0），且不再主动移动时，环境会因 `body_not_awake_or_settled` 而终止。这一条件可视为成功软着陆。
- failure-like termination:  
  - `crash_or_body_contact`：飞行器主体与地面、目标垫边缘或其他障碍物发生非支撑脚接触，代表坠毁。  
  - `horizontal_position_outside_viewport`：水平位置超出允许范围（掉出屏幕），代表失败。
- ambiguous termination:  
  - `body_not_awake_or_settled` 本身不区分成功与失败，若飞行器在错误位置或姿态下静止同样会触发该条件。必须结合位置、速度、接触情况才能判断成功与否。
- truncation: 本环境 step 返回的 truncated 恒为 False（`return ..., False, {}`），无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能假设存在 success、termination_reason 等键。

**结论**：只能从观测值与终止标志的组合自行推断成功/失败，无法直接读取环境内置的成功标识。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` (当前观测，一维数组长度为8)
- `action` (当前执行的动作编号)
- `next_obs` (执行动作后的观测)
- `info` 中**不存在**任何允许字段，禁止使用
- `training_progress` 在本次 prompt 未明确允许使用，禁止用于奖励

禁止使用：
- `original_reward` （正式环境 reward 已屏蔽）
- 任何名为 official_reward 或封装的原始奖励信号
- 未在上述允许列表中的 info 字段
- 超范围的 obs 切片（仅可使用 0~7 索引）

## 7. 可用于奖励函数的信号
以下信号均可安全用于奖励定义：
- position: `obs[0]`, `obs[1]`（目标垫相对坐标）
- velocity: `obs[2]`, `obs[3]`
- orientation: `obs[4]`, `obs[5]`
- contact: `obs[6]`, `obs[7]`
- action: `action` 值（当前动作编号，可用于燃料惩罚）
- next_obs 中对应的上述索引同样可用（如 `next_obs[0]` 等）
- 可以计算的辅助量：到目标垫的距离、速度幅值、姿态角误差等。

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：不存在，info 为空，不能依赖。
- 环境内部奖励：已被屏蔽，禁止使用。
- 接触之外的力学信息（如冲击力、滑动摩擦等）：观测中未提供，不可用。
- 任何未在观察空间中声明的维度。



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

## 1. 任务路由摘要
- multi_objective_task：按该任务类型选择信号，并先检查接口可用性。

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
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。