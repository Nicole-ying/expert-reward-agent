# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从初始位置（靠近视口顶部中心）尽快飞向中央的目标着陆垫，并在垫上稳定停靠。学习过程中要求尽量少使用主引擎，同时保持姿态平稳、安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x 坐标（相对于目标垫中心的水平距离）
- obs[1]: y 坐标（相对于目标垫高度的垂直距离）
- obs[2]: x 方向线速度
- obs[3]: y 方向线速度
- obs[4]: 机体角度（姿态朝向）
- obs[5]: 角速度
- obs[6]: 左侧支撑接触标志（0 或 1）
- obs[7]: 右侧支撑接触标志（0 或 1）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: 无引擎（不做任何推进）
- action 1: 左侧姿态引擎（启动左侧转向引擎，产生力矩）
- action 2: 主引擎（启动主推进引擎）
- action 3: 右侧姿态引擎（启动右侧转向引擎，产生反方向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
终止由以下任一条件触发（terminated = True）：
1. 机体碰撞地面或其他物体 crash_or_body_contact
2. 水平位置超出视口 horizontal_position_outside_viewport
3. 机体不再活跃或已稳定睡眠 body_not_awake_or_settled

- success-like termination: 无明显唯一成功标志；可能体满足“到达并在垫上稳定”时 body_not_awake_or_settled 为真，但也可能在其他位置提前睡眠，故不能直接当作成功。
- failure-like termination: crash_or_body_contact 和 horizontal_position_outside_viewport 可能为失败终止。
- ambiguous termination: body_not_awake_or_settled 可能对应成功着陆稳定，也可能对应非目标位置的过早睡眠。
- truncation: 无，truncated 始终为 False。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典）
- forbidden_or_uncertain_info_fields: info 无任何可用字段，不能从中提取成功/失败/原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前观测数组
- next_obs：下一时刻观测数组
- action：当前采取的动作索引
- info 中明确允许的字段（此处 info 为空，所以无可使用字段）

禁止使用：
- original_reward（官方奖励）
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片含义
- training_progress（除非后续说明允许，默认为禁止）

## 7. 可用于奖励函数的信号
- position: obs[0] 相对水平距离、obs[1] 相对垂直距离
- velocity: obs[2] 和 obs[3] 的线速度大小
- orientation: obs[4] 机体角度、obs[5] 角速度
- contact: obs[6]、obs[7] 左右支撑接触标志（着陆垫信号）
- action/engine: 动作选择，尤其可以惩罚主引擎使用（action == 2）

## 8. 不确定或不可用的信号
- 官方奖励 original_reward 被禁止，不可还原
- info 无内容，不能获得 success/failure 标志，也不能获得任何评估提示
- 终止原因无显式返回，无法从 step 返回值区分具体终止类型
- 无外界提供的速度/加速度参考值，需从观测自行推导奖励项



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