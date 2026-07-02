# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从一个随机初始状态尽快飞到中央的目标着陆垫，并稳定停留在其上。  
目标包括三个子要求：  
- 尽可能短的时间到达目标上方并降落；  
- 着陆时速度与姿态尽量平稳（接触安全）；  
- 整个飞行过程中尽量少用引擎推力（节省能量）。  

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是使飞行器到达并停靠在空间中的一个目标位置（着陆垫），属于典型的导航目标到达任务；能量消耗是次要优化指标，不改变任务本质。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 推断为 float64 或 float32（连续值，接触标志为 0.0/1.0）  
- 维度含义：  
  - obs[0] (x_position): 飞行器相对于目标着陆垫中心的水平坐标  
  - obs[1] (y_position): 飞行器相对于目标着陆垫高度的垂直坐标  
  - obs[2] (x_velocity): 飞行器水平线速度  
  - obs[3] (y_velocity): 飞行器垂直线速度  
  - obs[4] (body_angle): 飞行器机体角度（orientation）  
  - obs[5] (angular_velocity): 飞行器角速度  
  - obs[6] (left_support_contact): 左支撑腿接触标志（1.0 接触，0.0 未接触）  
  - obs[7] (right_support_contact): 右支撑腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：  
  - action 0: no_engine —— 不点火，依靠惯性滑行  
  - action 1: left_orientation_engine —— 点燃左侧姿态调整引擎（产生转动）  
  - action 2: main_engine —— 点燃主引擎（产生向上的推力）  
  - action 3: right_orientation_engine —— 点燃右侧姿态调整引擎（产生反向转动）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  - body_not_awake_or_settled（飞行器静止/稳定，可能表示成功着陆并稳定在垫子上）  
- failure-like termination:  
  - crash_or_body_contact（飞行器与地面或其他物体异常碰撞）  
  - horizontal_position_outside_viewport（水平方向飞出有效区域）  
- ambiguous termination:  
  - body_not_awake_or_settled 本身不携带位置信息，需结合 obs[0] 和 obs[1] 是否接近目标来判断真正的成功。  
- truncation:  
  - 始终为 False（无时间截断）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: [] （info 字典为空，无可信字段）  
- forbidden_or_uncertain_info_fields: 全部 info 字段均禁止使用（不存在任何可信标志）

## 6. reward 函数接口契约
函数签名必须为：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察向量）
- action（当前执行的动作）
- next_obs（下一时刻观察向量）
- info 中明确允许的字段（当前允许列表为空）
- training_progress（仅在 prompt 明确允许时使用）

禁止使用：
- original_reward（来自环境的原始奖励）
- official_reward（任何形式的环境内部奖励）
- 未声明的 info 字段（即所有 info 键）
- 未声明的 obs 切片（不得滥用未说明意义的维度）

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，可得到距离目标垫的距离  
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，可用于惩罚硬着陆或大速度  
- orientation: obs[4] (body_angle)，可鼓励保持在直立姿态  
- angular_velocity: obs[5] (angular_velocity)，可惩罚剧烈旋转  
- contact: obs[6] (left_contact), obs[7] (right_contact)，可检测着陆腿是否均已安全接触  
- action/engine: 动作本身就是燃料消耗信号，可惩罚非零推力的动作以鼓励节能

## 8. 不确定或不可用的信号
- original_reward / official_reward —— 强制屏蔽，不可用  
- info 内任何字段（如 success 标志） —— 不可用（info 为空）  
- 目标垫是否被准确接触（需要结合位置与接触信号自行判断）  
- 是否发生 crash 等事件的具体标志（只能通过观察变化和终止原因推断）  
- 外部风扰等环境未知因素 —— 不可观测



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