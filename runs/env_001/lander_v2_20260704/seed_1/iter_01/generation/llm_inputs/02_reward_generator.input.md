# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
该环境是一个 2D 飞行器/载具轨迹优化任务。一个主体从视口顶部中央区域出发并受到随机初始力。目标是**尽快到达并稳定降落在视口中央的目标着陆垫上**，同时**尽量少使用引擎推力**。智能体需要学习接近目标、减速、保持稳定姿态并安全触地（通过左右支撑腿接触），而不是主体碰撞。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 默认 float32  
- obs[0] (x_position): 水平坐标（相对于目标垫的位置，负表示在左，正在右）  
- obs[1] (y_position): 垂直坐标（相对于目标垫高度的位置，负表示低于垫，正表示高于垫）  
- obs[2] (x_velocity): 水平线速度  
- obs[3] (y_velocity): 垂直线速度  
- obs[4] (body_angle): 主体朝向角（弧度或其它单位）  
- obs[5] (angular_velocity): 角速度  
- obs[6] (left_support_contact): 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触）  
- obs[7] (right_support_contact): 右支撑腿接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete(4)  
- action 0: no_engine —— 不执行任何引擎推力  
- action 1: left_orientation_engine —— 点燃左姿态引擎（提供旋转力矩）  
- action 2: main_engine —— 点燃主引擎（提供主要推力，通常朝下或沿主体方向）  
- action 3: right_orientation_engine —— 点燃右姿态引擎（提供反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` —— 主体停止运动且稳定。结合任务目标，这通常意味着已落在垫上并稳定，可视为成功着陆。  
- failure-like termination:  
  `crash_or_body_contact` —— 主体与其他表面发生不安全碰撞（区别于支撑腿安全接触）。  
  `horizontal_position_outside_viewport` —— 水平位置超出视口边界。  
- ambiguous termination: 无。注意：`body_not_awake_or_settled` 在极少数异常情况下（如卡在空中）才可能不是成功，但通常环境设计使其与成功着陆关联。  
- truncation: 无（该环境使用终止而非截断，info 为空且未提及 max steps）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 为空字典 {}，不允许推断或在此之上构造信号）  
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用，因为 info 中不包含 success、failure 或 termination_reason 等字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察的 8 维数组）
- action（执行的离散动作编号 0~3）
- next_obs（下一观察的 8 维数组，即使 terminated=True 也提供最终状态）
- info 中明确允许的字段（当前无）
- training_progress 仅在 prompt 明确允许时可使用（本环境卡片未提及，故禁止）

禁止使用：
- original_reward（被环境屏蔽，不得访问或尝试重现）
- official_reward
- 未声明的 info 字段（例如 success、failure、termination_reason 等）
- 未声明的 obs 切片（必须严格使用 obs[0]~obs[7] 的含义）

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) 及对应的 next_obs 值  
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) 及 next_obs  
- orientation: obs[4] (body_angle), obs[5] (angular_velocity)  
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) —— 表示着陆腿是否安全触地  
- action/engine: 当前动作 action（0~3），可用于惩罚引擎使用（如 action 不为 0 时惩罚推力）

## 8. 不确定或不可用的信号
- 官方奖励 original_reward 被屏蔽，无法使用  
- 终止原因（crash / out of viewport / settled）没有通过 info 暴露，不能直接作为奖励信号  
- 无法获得显式的成功/失败标志（如 info["success"] 不存在）  
- 只能通过 next_obs 的状态（位置、速度、接触、角度）间接推断是否成功着陆并稳定；但也因此无法 100% 确定终止类型，只能基于领域知识推断（如在接近目标垫且双腿接触、速度很小时终止可视为成功）



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