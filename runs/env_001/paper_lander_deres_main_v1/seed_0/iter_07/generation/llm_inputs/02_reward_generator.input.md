# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器样式的轨迹规划任务。飞行器初始时位于画面上方中央区域，并受到随机的初始力。目标是在尽可能短的时间内飞抵画面中央的目标垫并稳定停靠，同时尽量少使用主引擎或方向引擎推力。智能体需要学会接近目标地点、减小速度、保持平稳姿态，并以安全接触的方式着陆到目标垫上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，Box 默认为 float）
- obs[0]: x_position — 飞行器相对于目标垫中心的水平坐标
- obs[1]: y_position — 飞行器相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 飞行器水平方向线速度
- obs[3]: y_velocity — 飞行器垂直方向线速度
- obs[4]: body_angle — 飞行器机身的姿态角（弧度）
- obs[5]: angular_velocity — 飞行器角速度
- obs[6]: left_support_contact — 左侧支撑腿是否接触目标垫（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑腿是否接触目标垫（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 什么也不做，不施加推力
- action 1: left_orientation_engine — 启动左侧方向引擎（用于调整姿态）
- action 2: main_engine — 启动主引擎（产生推力，改变速度）
- action 3: right_orientation_engine — 启动右侧方向引擎（用于调整姿态）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（休眠/稳定），当飞行器在目标垫附近且姿态稳定时很可能表示成功着陆并稳定停留
- failure-like termination: crash_or_body_contact（机身非着陆部位发生碰撞）；horizontal_position_outside_viewport（水平位置超出画面边界）
- ambiguous termination: body_not_awake_or_settled 也可能在远离目标垫的位置发生，例如因撞击后静止，因此不能直接等同于成功，需要结合位置信息判断
- truncation: 未在源文件中出现，但一般由时间步上限引发

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 固定为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用（包括任何可能隐含的终止原因字段）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测
- action：当前执行的动作
- next_obs：下一步观测
- info 中明确允许的字段（当前无任何字段被明确允许）
- training_progress（仅当提示明确要求使用时才可安全使用，否则应视为不可用）

禁止使用：
- original_reward（原始奖励信号已被屏蔽，严禁依赖）
- official_reward 或任何形式的官方奖励
- 未声明的 info 字段（info 完全为空，使用任何字段均违背契约）
- 未在环境卡片中声明的 obs 切片含义

## 7. 可用于奖励函数的信号
- position: obs[0]（水平位置相对目标）、obs[1]（垂直相对高度）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（姿态角）
- angular velocity: obs[5]
- contact: obs[6]（左支撑腿接触）、obs[7]（右支撑腿接触）
- action/engine: 动作编号本身可反映使用何种引擎，用于衡量推力使用量

## 8. 不确定或不可用的信号
- info 字典：完全为空，无法提供任何额外终止原因或成功标记
- original_reward：已被屏蔽，数值不可知，不允许读取或拟合
- 机身碰撞详情：crash_or_body_contact 的具体碰撞类型在观测中不可见
- 全局时间步或剩余时间：未在观测中提供
- 目标垫绝对坐标：只能通过相对位置推知接近程度



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



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- approach_quality_reward + attitude_penalty + contact_bonus + progress_reward
- approach_quality_reward + attitude_penalty + landing_proxy + progress_reward
- approach_quality_reward + attitude_penalty + progress_reward
- attitude_penalty + landing_quality_reward + progress_reward

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
