# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个 2D 车辆着陆任务：主体从视口顶部中心区域以随机初始推力出发，需尽快到达并稳定停留在中央目标着陆台，同时尽量减少主发动机和姿态发动机的使用。智能体需要学会靠近目标、减速、保持正确朝向、安全接触着陆腿，避免主体硬接触地面或飞出边界。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务明确要求“到达并停留在中央目标着陆台”，核心是导航至目标位置并稳定着陆，属于经典的目标到达型任务；虽然同时涉及速度控制与姿态稳定，但最终目标仍是到达指定垫子，因此归入 navigation_goal_reaching。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 默认 float64 或 float32（通常为 float32，视具体实现而定）  
- 各维度含义：
  - obs[0]：x_position —— 相对于目标台的水平坐标（target pad 中心为 0）  
  - obs[1]：y_position —— 相对于目标台垫面高度的垂直坐标  
  - obs[2]：x_velocity —— 水平线速度  
  - obs[3]：y_velocity —— 垂直线速度  
  - obs[4]：body_angle —— 主体朝向角度  
  - obs[5]：angular_velocity —— 角速度  
  - obs[6]：left_support_contact —— 左支撑腿接触标志（0.0 未接触，1.0 接触）  
  - obs[7]：right_support_contact —— 右支撑腿接触标志（0.0 未接触，1.0 接触）

## 4. 动作空间 action_space
- type: Discrete  
- 动作数量：4  
- 每个动作 ID 含义：
  - 0 (no_engine)：不点火，不施加推力  
  - 1 (left_orientation_engine)：点燃左姿态发动机，产生向右旋转力矩（假定）  
  - 2 (main_engine)：点燃主发动机，产生向上推力（大致沿主体方向）  
  - 3 (right_orientation_engine)：点燃右姿态发动机，产生向左旋转力矩（假定）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - body_not_awake_or_settled 且至少一侧支撑腿接触地面（可能为成功软着陆），即主体稳定后自动判定为终止。该条件可能包含成功着陆，但需结合接触标志确认。  
- failure-like termination:
  - crash_or_body_contact：主体发生碰撞或硬接触（非着陆腿），视为坠毁失败。  
  - horizontal_position_outside_viewport：水平位置超出视口边界，视为飞出、失败。  
- ambiguous termination:
  - body_not_awake_or_settled 在无支撑腿接触时可能表示主体已静止但未接触垫子（例如停在悬崖外），此时是否为成功需要进一步判断，但因信息缺失暂归为模糊终止。  
- truncation:
  - 当前 step 源中未出现时间截断，仅以三个条件终止，无 max_episode_steps。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 返回空字典 {}，可安全使用信息字段极少）  
- forbidden_or_uncertain_info_fields: 所有 info 字段，因为源代码明确返回空字典，任何未声明的键（如 "success"、"failure"）均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前观察的 8 维数组，可用于提取位置、速度、角度、接触等信号  
- action：执行的动作 ID（0~3）  
- next_obs：下一时刻观察，用于计算变化量  
- info：当前允许的字段为空，可忽略  
- training_progress：仅当 prompt 明确允许时使用（本例未声明允许，故不可用）

禁止使用：
- original_reward：原始奖励被屏蔽，严禁访问  
- official_reward：同义，禁止使用  
- 未声明的 info 字段（如 "success"、"termination_reason"）  
- 未在观察空间中明确定义的切片或含义

推荐基于 obs 与 next_obs 构建进度、能耗、稳定性等信号，不使用原始奖励。

## 7. 可用于奖励函数的信号
基于可用的观测维度与动作：

- position: obs[0] (X 相对目标), obs[1] (Y 相对台高) —— 可衡量靠近目标的距离及高度差  
- velocity: obs[2], obs[3] —— 线速度，结合接触或接近时可评估着陆平稳性  
- orientation: obs[4] —— 主体角度，用于惩罚大幅倾斜  
- angular_velocity: obs[5] —— 角运动状态，可奖励稳定  
- contact: obs[6], obs[7] —— 支撑腿接触标志，可确认着陆并奖励双腿着地  
- action/engine: action 值 —— 可施加推力的惩罚，鼓励最小能源使用

所有上述信号均可安全使用，因为它们直接来自观察空间，且与任务目标高度相关。

## 8. 不确定或不可用的信号
- 原始奖励（被屏蔽）  
- info 内任何字段（空字典）  
- explicit success/failure flag（不存在）  
- 终止原因的文本标签（不存在）  
- 风速、推力等内部物理量（未暴露）  
- 是否完全止动或依然活跃的明确标志（body_not_awake_or_settled 仅在 step 内用于终止判定，未出现在观察中，不可在奖励函数中使用）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- navigation_goal_reaching：用密集过程引导；无 success flag 时禁用终点成功核心项；重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要
### progress_delta_reward
- 角色: 主学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 本轮建议: 推荐作为 v1 主信号：奖励每一步更接近目标。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 本轮建议: 可作为小权重 anchor；不要和 progress_delta_reward 同时大权重堆叠。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 本轮建议: 如果任务要求稳定接近/着陆，v1 可以小权重加入。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速撞击或姿态失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 本轮建议: 可选小权重；不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 本轮建议: 若 explicit_success_flag_available=false，不作为 v1 核心项。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 本轮建议: 若 explicit_failure_flag_available=false，不作为 v1 核心项。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 本轮建议: 通常后续加入，不建议 v1 太早加入。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 本轮建议: 通常后续加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能接近并稳定后再优化燃料。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 本轮建议: v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 本轮建议: 不作为 v1 首选；比 progress_delta 更抽象。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget，而不是固定组件数量。
- 推荐 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。
- 如果速度/姿态信号明确可用，且任务需要稳定接近或着陆，可以加入轻量 stability_penalty。
- 如果使用 contact，只能作为 soft_landing_proxy 的一部分，必须和 near_target、low_speed、stable_angle 组合，不要直接把 contact 当 success。
- energy_penalty、time_penalty、gated_reward 默认后续迭代再加入。
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。