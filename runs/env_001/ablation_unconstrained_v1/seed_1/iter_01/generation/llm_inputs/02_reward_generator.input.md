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