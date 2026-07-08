# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器（或类似着陆器）轨迹优化任务。飞行器从视野上方偏向中心的位置出发，带有随机的初始作用力。目标是**尽快降落到中央的目标平台上，并在此过程中保持姿态稳定、速度轻微，实现安全着陆**，同时**尽可能少地使用主发动机推力**。换句话说，核心是到达指定平台并安全停稳，附属优化是减少燃料消耗和快速完成。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 推断为 float32（位置/速度连续值，接触标志为 0.0/1.0 浮点）
- obs[0] (x_position): 飞行器水平坐标，相对于目标平台中心的偏移量
- obs[1] (y_position): 飞行器垂直坐标，相对于平台表面的高度
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机身朝向角度（rad 或归一化角度）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑接触标志，接触时为 1.0，否则 0.0
- obs[7] (right_support_contact): 右侧支撑接触标志，接触时为 1.0，否则 0.0

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0 (no_engine): 不启动任何发动机，飞行器仅受当前动量和重力/物理影响
- action 1 (left_orientation_engine): 启动左侧姿态发动机，产生旋转力矩（改变角速度/角度）
- action 2 (main_engine): 启动主发动机，产生向上的推力（影响 y 方向速度）
- action 3 (right_orientation_engine): 启动右侧姿态发动机，产生相反的旋转力矩

## 5. step 与终止条件分析

### 5.1 终止模式
根据代码和描述，终止条件为以下三种逻辑值的 **或**：
- **crash_or_body_contact**：飞行器主体发生碰撞（可能包括猛烈着陆或超出安全阈值的接触）
- **horizontal_position_outside_viewport**：水平位置超出视野范围
- **body_not_awake_or_settled**：飞行器“休眠”或已稳定停住（物理引擎判定速度/角速度极小）

具体成功/失败对应关系如下：
- success-like termination: 仅在 **body_not_awake_or_settled** 发生时，且同时满足**所有接触腿恰好着陆在平台上、速度/姿态处于安全范围内**的场景（但代码中未显式给出“安全”判定，仅能从任务描述和目标推断）。通常成功停稳后才触发该终止。
- failure-like termination: **crash_or_body_contact**（如果着陆时速度过高或姿态错误而导致碰撞）或 **horizontal_position_outside_viewport** 都会导致失败终止。需要注意的是，crash_or_body_contact 也可能包含“安全着陆后仍在接触状态”的情形，但由于安全着陆后很快就会进入 not_awake 状态，所以成功路径通常不会由该条件单独触发；但单纯基于终止条件的划分，**仅有 not_awake 而没有任何接触也可能属于成功（接触腿可能在平台之外的不安全情况？）**，这需要额外信息，因此直接使用终止条件判断成功与否并不完全可靠。
- ambiguous termination: 仅凭 terminated=true 和三个原始条件无法可靠区分成功与失败，因为**没有显式的 success 或 failure 标志**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典 `{}`，没有任何 success 字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 禁止使用任何 info 字段（如 info["success"]、info["failure"]、info["termination_reason"]）——它们不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：动作执行前的 8 维观测（numpy 数组或类似容器）
- `action`：执行的离散动作 id（int）
- `next_obs`：动作执行后的 8 维观测
- `info` 中存在的字段：当前 info 为空字典，**没有任何可用字段**（不允许假设添加）
- `training_progress`：除非明确提示允许使用，否则**不应使用**

禁止使用：
- `original_reward`（官方奖励已被屏蔽，禁止利用其还原或计算）
- 任何未在观测空间中声明的隐式信号（如内部碰撞类型、燃料量等）
- 未明确提供的 info 字段（包括但不限于 success, failure, termination_reason 等）

## 7. 可用于奖励函数的信号
- **position**: `obs[0]` (x 相对位置)、`obs[1]` (y 相对高度) 以及相应 `next_obs` 值
- **velocity**: `obs[2]` (vx)、`obs[3]` (vy) 以及变化值
- **orientation**: `obs[4]` (机体角度)、`obs[5]` (角速度)
- **contact**: `obs[6]` (左腿接触)、`obs[7]` (右腿接触) —— 可用于判断着陆状态
- **action/engine**: `action` 值可用于惩罚主发动机使用（action == 2）或姿态发动机使用，以鼓励节省燃料

## 8. 不确定或不可用的信号
- 官方原始奖励 (`original_reward`)：完全禁止使用
- 碰撞严重程度或碰撞类型：观测中无法获得（虽有 crash_or_body_contact 触发终止，但未传入观测或 info）
- 燃料剩余量：无相关观测
- 目标平台中心是否存在标志符：无
- 成功标志或失败原因（如是否软着陆、可着陆区域是否合法）：info 为空，禁止使用
- 任务的真实名称或已知基准（Lunar Lander 等）：禁止猜测或使用任何外部先验



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



