# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
环境模拟一个 2D 飞行器轨迹优化问题。一个刚体从视口顶部中心附近出发，带有随机初始扰动力。目标是**尽可能快地飞到中央目标平台并平稳降落**，同时尽量少用引擎推力。智能体需要学会靠近目标、减小速度、保持稳定姿态，并安全接触平台。

简化为一句话：以最小的燃料消耗和最短时间，精准降落到固定的目标平台上并稳定下来。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float64 或 float32 (具体取决于底层实现，均为浮点)
- obs[0]: x_position —— 飞行器水平位置相对于目标平台的水平偏移
- obs[1]: y_position —— 飞行器垂直位置相对于平台高度
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 机体旋转角速度
- obs[6]: left_support_contact —— 左侧支撑点是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右侧支撑点是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃左侧姿态调节发动机（产生旋转力矩）
- action 2: main_engine —— 点燃主发动机（产生与机体角度相关的推力，通常用于减速/悬浮）
- action 3: right_orientation_engine —— 点燃右侧姿态调节发动机（产生相反旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` —— 机体速度与角速度极低且可能已接触，视为稳定着陆。
- failure-like termination:  
  `horizontal_position_outside_viewport` —— 水平位置超出边界，飞行器丢失或失控。
- ambiguous termination:  
  `crash_or_body_contact` —— 字面上包含“撞击”和“身体接触”，可能包含不安全的碰撞（失败）或刚好接触平台但未稳定（失败）甚至安全接触（成功）。由于无法从返回值直接区分，除非结合其他状态判断，否则不可作为干净的成败信号。
- truncation:  
  无。step 返回元组中 truncated 恒为 False，说明没有时间步上限截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 恒为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 当前观察（8 维数组）
- action: 执行的离散动作（0~3）
- next_obs: 下一时刻观察（8 维数组）
- info 中明确允许的字段：无（info 恒为空，禁止访问其中任何字段）
- training_progress: 仅当 prompt 明确允许时才可使用，本次未授权

禁止使用：
- original_reward: 已被掩码的官方奖励，严禁读取或模仿
- official_reward: 同 original_reward
- 未声明的 info 字段
- 未声明的 obs 切片含义（只能使用本卡片第 3 节列出的索引含义）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]) 和 y_position (obs[1])，可计算到目标的距离
- velocity: x_velocity (obs[2]) 和 y_velocity (obs[3])，可评估降落平稳性
- orientation: body_angle (obs[4]) 和 angular_velocity (obs[5])，可衡量姿态稳定性
- contact: left_support_contact (obs[6]) 和 right_support_contact (obs[7])，可判断是否已经接触、是否双足着地
- action/engine: 当前动作是否使用引擎（0为非零推力，2为主推，1、3为姿态推），可评估燃料消耗

## 8. 不确定或不可用的信号
- original_reward: 被彻底掩码，不可用
- info 字典内任何信号：不可用（因为恒为空）
- crash_or_body_contact 的准确含义：无法区分成功/失败，不能作为直接的奖励来源，仅可在终止时结合其他状态推测
- 外界风速或其他干扰力：在 step 源代码中被省略，无法确定其存在及影响



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