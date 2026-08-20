# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
该环境是一个二维飞行器着陆任务。  
智能体控制飞行器从顶部中央区域出发，受初始随机力影响，需在最短时间内到达并稳定在中央目标垫上。  
过程中必须尽量降低推进剂消耗（减少引擎使用），同时学习精确导航、减速、保持姿态并安全触垫。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务明确要求到达中心目标垫并稳定，具有明确的导航目标位置（坐标原点），符合目标导向的导航类任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（接触标志以 1.0/0.0 形式出现，推断为浮点型）
- obs[0]: x_position —— 飞行器相对于目标垫中心的水平坐标
- obs[1]: y_position —— 飞行器相对于目标垫高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体俯仰/倾斜角度
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左侧支撑腿/结构接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact —— 右侧支撑腿/结构接触标志（同上）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不开启任何引擎，惯性滑行
- action 1: left_orientation_engine —— 点燃左姿态引擎，产生旋转力矩（姿态调整）
- action 2: main_engine —— 点燃主引擎，提供垂直方向的推力（主要用于减速/上升）
- action 3: right_orientation_engine —— 点燃右姿态引擎，产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 飞行器停止运动且稳定（通常意味着已在目标垫上安全着陆）
- failure-like termination: 
  - crash_or_body_contact —— 飞行器主体碰撞地面或发生硬接触（非正常着陆）
  - horizontal_position_outside_viewport —— 水平位置超出可视边界
- ambiguous termination: 无
- truncation: 未在任务说明中明确给出，但环境中通常存在最大时间步上限（truncation），此处不依赖其信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  （info 字典为空，无 `success` 字段）
- explicit_failure_flag_available: false  
  （同理，无 `failure` 或 `termination_reason`）
- allowed_info_fields: {} （info 为空，不应使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 内容均不可用，禁止依赖任何隐含的终止原因。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（8 维数组）
- action（整数 0-3）
- next_obs（8 维数组）

禁止使用：
- original_reward（已被屏蔽，不可复原）
- official_reward 或任何形式的真实 reward
- info（空字典，无任何字段可用）
- training_progress（仅当环境 prompt 明确允许时才可使用，本环境未允许）

## 7. 可用于奖励函数的信号
以下信号可直接从 obs 或 next_obs 中提取，可用于设计奖励：
- position： obs[0]（x 位置）、obs[1]（y 位置），均以目标垫为参考系
- velocity： obs[2]（x 速度）、obs[3]（y 速度）
- orientation： obs[4]（机体角度）、obs[5]（角速度）
- contact： obs[6]（左接触）、obs[7]（右接触）
- action/engine： action 0-3，可用于判断是否使用引擎，施加燃料消耗惩罚

## 8. 不确定或不可用的信号
- original_reward：不可用
- info 中的任何信号（例如 success、failure、termination_reason）：info 为空，不可用
- 明确的“已到达目标”标志：环境没有提供直接的 reach flag，需要通过位置/速度/接触组合自行判断（例如 next_obs[1]≈0 且接触标志为真、速度很小可视为成功着陆）
- 外部施加的风或随机力：仅初始存在，后续 step 中未给出观测，不可直接使用（但可通过速度变化间接体现）



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



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- contact_bonus + potential_shaping
- distance_reward + soft_landing + stability_penalty
- distance_reward + soft_landing_continuous + stability_penalty
- distance_reward + soft_landing_proxy + stability_penalty

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
