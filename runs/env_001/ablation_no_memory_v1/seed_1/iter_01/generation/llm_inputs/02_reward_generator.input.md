# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D类车辆轨迹优化任务。一个主体从视口顶部中央附近开始，带有随机初始力。目标是**尽快到达并稳定在中央目标垫上**，同时**尽可能少地使用引擎推力**。智能体需要学会靠近目标、减速、保持稳定姿态并安全着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position — 相对于目标垫的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 身体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑接触标志（0.0 或 1.0）
- obs[7]: right_support_contact — 右支撑接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不施加任何引擎推力
- action 1: left_orientation_engine — 启动左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 启动主引擎（产生主要推力）
- action 3: right_orientation_engine — 启动右侧姿态引擎（产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
环境在以下任一条件成立时终止（terminated = True）：
- `crash_or_body_contact` — 发生撞击或非安全接触
- `horizontal_position_outside_viewport` — 水平坐标超出视口范围
- `body_not_awake_or_settled` — 身体进入休眠状态或完全静止

分析：
- 前两条通常对应失败（撞击、飞出边界）。
- `body_not_awake_or_settled` 含义模糊：当主体在目标垫上成功稳定着陆时，也会触发静止（成功情况）；但主体可能在错误位置停止或卡住，同样触发。因此该终止**不能直接区分成功与失败**。
- 所有终止信息仅合并为一个布尔值 `terminated`，无额外解释字段。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: info 的所有字段（不可用），original_reward，任何未在 observation 中声明的信号

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` — 当前观测向量
- `action` — 当前动作索引
- `next_obs` — 下一步观测向量
- 无

禁止使用：
- `original_reward`（官方奖励已掩码，不得还原）
- `info` 的任何字段（info 为空，且不允许假设其存在字段）
- `training_progress`（本环境说明中未允许使用）
- 任何未在 observation_space 中明确声明的观测切片或计算

## 7. 可用于奖励函数的信号
以下信号完全从 `obs`、`action`、`next_obs` 中可获取：
- **位置信号**：x_position, y_position（相对于目标垫），可用于度量到目标的距离
- **速度信号**：x_velocity, y_velocity，可用于惩罚过快运动或鼓励减速
- **姿态信号**：body_angle, angular_velocity，可用于鼓励保持稳定角度（例如竖直）
- **接触信号**：left_support_contact, right_support_contact，可反映着陆脚是否着地、是否平稳
- **动作/引擎信号**：action 选择（0~3），可用于惩罚非零动作以节省燃料

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不可用，终止逻辑中不提供 success/failure 标记
- **终止原因解析**：无法准确得知是因为安全着陆还是异常停止而终止
- **官方原始奖励**：被屏蔽，禁止使用
- **任何 info 字段**：空字典，不可用
- **真实环境名或基准信息**：已匿名化，不可使用



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