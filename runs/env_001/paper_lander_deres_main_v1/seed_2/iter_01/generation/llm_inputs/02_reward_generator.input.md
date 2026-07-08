# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
一个 2D 飞行器从画面顶部中央出发，受随机初始力影响。核心目标是让飞行器尽快到达画面中央的目标着陆垫并稳定停稳，同时尽量减少引擎推力使用。智能体需要学习靠近目标、减速、保持姿态稳定并实现安全着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: 飞行器相对于目标垫中心的水平坐标 (x_position)
- obs[1]: 飞行器相对于目标垫高度的垂直坐标 (y_position)
- obs[2]: 水平线速度 (x_velocity)
- obs[3]: 垂直线速度 (y_velocity)
- obs[4]: 机身姿态角 (body_angle)
- obs[5]: 角速度 (angular_velocity)
- obs[6]: 左支撑腿接触标志，接触为 1.0，否则为 0.0 (left_support_contact)
- obs[7]: 右支撑腿接触标志，接触为 1.0，否则为 0.0 (right_support_contact)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎推力 (no_engine)
- action 1: 点燃左侧姿态引擎 (left_orientation_engine)
- action 2: 点燃主引擎 (main_engine)
- action 3: 点燃右侧姿态引擎 (right_orientation_engine)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器停止活动并稳定，可能表示成功着陆并停稳）
- failure-like termination: crash_or_body_contact（撞击或非正常身体接触），horizontal_position_outside_viewport（水平出界）
- ambiguous termination: body_not_awake_or_settled 在未接触目标垫或错误位置时可能不表示真正成功
- truncation: 未明确提及，假设无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info为空字典）
- forbidden_or_uncertain_info_fields: info（空字典，不可提供任何额外信号）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（全部8维）
- action
- next_obs（全部8维）
- info 中明确允许的字段（当前无允许字段）
- training_progress 仅在 prompt 明确允许时使用，此处默认禁止

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片（只能使用观测的原始含义和完整索引）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])，可用于衡量与目标垫的距离
- velocity: x_velocity (obs[2]), y_velocity (obs[3])，可用于衡量降落时的速度大小
- orientation: body_angle (obs[4]), angular_velocity (obs[5])，可用于姿态稳定
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])，可用于判断着陆
- action/engine: 动作选择本身（主引擎或姿态引擎），可用于惩罚推力使用

## 8. 不确定或不可用的信号
- 明确的成功/失败标志：info 中无任何字段，无法直接获得奖励用的 ground truth
- 燃料消耗量：观测中无直接燃料读数，仅能通过动作间接推测
- 目标垫是否被准确瞄准或是否完成软着陆：仅可通过位置、速度和接触组合判断，无显式事件标记



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

## 1. 任务路由摘要
- navigation_goal_reaching：任务目标是接近/到达指定位置。重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

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

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

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