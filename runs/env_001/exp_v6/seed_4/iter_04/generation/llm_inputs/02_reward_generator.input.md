# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
本环境为一个 2D 飞行器轨迹优化任务。智能体初始位于视口顶部中央附近，受到随机初始力。目标是尽快到达中央目标垫（着陆平台）并稳定着陆，同时尽可能节省引擎推力。智能体需要学会接近目标、减小速度、保持稳定姿态并安全接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是引导飞行器从初始位置到达并稳定在目标点（中央目标垫），典型的导航目标到达问题；同时存在引擎燃料节省、姿态稳定等辅助目标，但仍以位置收敛为主。

## 3. 观察空间 observation_space
- type: Box (连续向量)
- shape: (8,)
- dtype: float64
- obs[0]: x_position —— 相对于目标垫的水平坐标（越接近 0 越对准目标）
- obs[1]: y_position —— 相对于着陆平台顶面的垂直坐标（越接近 0 越贴近平台）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体倾斜角度（弧度）
- obs[5]: angular_velocity —— 机体角速度
- obs[6]: left_support_contact —— 左支撑脚接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右支撑脚接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无操作（no_engine）—— 不启用任何引擎，仅凭惯性飞行
- action 1: 左姿态引擎（left_orientation_engine）—— 启动一个方向姿态调整引擎
- action 2: 主引擎（main_engine）—— 启动主引擎产生推力（通常向上减速）
- action 3: 右姿态引擎（right_orientation_engine）—— 启动相反方向姿态调整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  当满足 body_not_awake_or_settled（机体静止/休眠）且可能同时位于目标垫上方/接触时，视为成功稳定着陆。但终止条件中**没有明确判定是否在目标垫上**，仅凭“静止”无法保证成功，需要结合位置和接触信息推测。
- failure-like termination:  
  - crash_or_body_contact —— 机体发生非期望碰撞（如与地面、非目标物体接触）  
  - horizontal_position_outside_viewport —— 水平位置越界（掉出视口）
- ambiguous termination:  
  body_not_awake_or_settled 若发生在非目标位置或未接触平台，可能属于失败（如悬停后坠落或停在边缘），但终止条件本身混合了成功与失败语义。
- truncation:  
  仅当达到最大步数时由环境外部截断（本环境 step 返回 `False` 给 truncation，说明不会主动截断，但可能由上层框架限制步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []  （info 字典为空，无任何可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用，尤其是不可依赖 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等常见但未提供的键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 当前观察向量（8维）
- action —— 当前执行的动作（0-3）
- next_obs —— 下一时刻观察向量
- info 中明确允许的字段（当前无允许字段）
- training_progress —— 仅在 prompt 明确要求使用时可用，本环境卡片不强制使用

禁止使用：
- original_reward —— 被屏蔽的原始奖励，严禁直接或间接拷贝
- official_reward —— 同上
- 未声明的 info 字段
- 未声明的 obs 切片解释（如自行猜测其他物理量）

## 7. 可用于奖励函数的信号
- position: obs[0] (相对目标水平距离), obs[1] (相对平台高度) —— 可用于引导靠近目标垫
- velocity: obs[2] (水平速度), obs[3] (垂直速度) —— 可用于鼓励减速、平稳着陆
- orientation: obs[4] (机体倾斜角) —— 可用于鼓励保持垂直姿态
- angular_velocity: obs[5] —— 可用于惩罚剧烈旋转
- contact: obs[6] (左脚接触), obs[7] (右脚接触) —— 可用于判断着陆成功并给予正奖励
- action/engine: action == 2（主引擎）可用于惩罚燃料消耗；action == 1 或 3 可用于惩罚过度姿态调整

## 8. 不确定或不可用的信号
- 原始奖励（被屏蔽）
- 任何显式成功/失败标志（info 为空）
- 目标垫接触以外的碰撞类型（crash_or_body_contact 无法细分为“致命碰撞”和“安全接触”，因终止条件未给出详细分类）
- 剩余步数或时间信息



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



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- ---
- progress + soft_landing_proxy + stability_penalty
- skeleton

请选择一个**完全不同的主信号骨架**。例如如果上述列表都是 progress_delta 系列，请尝试 potential_based_shaping 或 bounded_proximity。不要重复已失败的骨架。
