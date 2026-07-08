# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
一个二维飞行器从视口上方初始位置出发，受随机初始力。目标是 **尽快飞抵中央目标着陆点**，并在其上 **平稳停靠**（速度降至零、保持竖直姿态、双腿触地），同时尽可能 **少用主引擎推力** 以节省燃料。朝向、接触、越界等都会影响终止。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（位置速度等为连续值，接触为 0/1，但整体为浮点）
- obs[0]: x_position — 相对于目标着陆点中心的水平坐标
- obs[1]: y_position — 相对于目标着陆点高度平面的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体倾角（弧度制，推测）
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑点/腿接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑点/腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不开启任何引擎，滑行
- action 1: left_orientation_engine — 开启左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 开启主引擎（产生向上的推力）
- action 3: right_orientation_engine — 开启右侧姿态引擎（产生相反方向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: body_not_awake_or_settled 且满足成功条件（靠近目标、低速、竖直、双腿触地）——即平稳停靠目标上。
- **failure-like termination**: crash_or_body_contact（机体或关键部位强烈撞击地面/结构），horizontal_position_outside_viewport（水平飞出边界）。
- **ambiguous termination**: body_not_awake_or_settled 但未完全满足目标位置或接触条件（可能悬空失速、摔倒等），需根据上下文判断。
- **truncation**: 本环境未明确提及最大步数截断，但通常 RL 框架有步数上限，属于截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 返回空字典 {}）
- forbidden_or_uncertain_info_fields: info 中任何字段均不可用（不存在 success、failure、termination_reason 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`（当前观察）
- `action`（刚执行的动作）
- `next_obs`（下一时刻观察）
- `info` 为空，所以实际上没有可用 info 字段
- `training_progress` 只有在明确需要时才使用，此处任务未要求，故不应依赖

禁止使用：
- `original_reward`（官方奖励，严禁读取）
- `official_reward` 或其别名
- 未在观察空间中声明的 `obs` 切片
- 未明确允许的 `info` 字段（均不允许）

## 7. 可用于奖励函数的信号
- **position**: next_obs[0]（x 向目标距离）, next_obs[1]（y 向目标高度差）
- **velocity**: next_obs[2]（水平速度）, next_obs[3]（垂直速度）
- **orientation**: next_obs[4]（倾角）, next_obs[5]（角速度）
- **contact**: next_obs[6]（左腿触地）, next_obs[7]（右腿触地）
- **action/engine**: 使用 action == 2（主引擎）可计入燃料惩罚

## 8. 不确定或不可用的信号
- 是否成功着陆无显式标记，必须结合终止时刻的 `next_obs` 自行判断（如距离小、速度低、竖直、双腿触地）。
- 越界、坠毁等失败终止也只能从位置、速度突变或终止时刻的位置推断，但无法直接获得“crash”标志。
- 无法获得“是否静止”的直接信号，但可通过速度大小和角速度推断。
- 无 `info` 字段可用，无法获得任何环境内部度量（如剩余燃料、当前冲击力等）。



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
- approach_quality_reward + attitude_penalty + contact_reward + progress_reward
- approach_quality_reward + attitude_penalty + landing_proxy + progress_reward
- approach_quality_reward + attitude_penalty + progress_reward
- attitude_penalty + descent_incentive + landing_settle + progress_reward
- attitude_penalty + landing_quality_reward + progress_reward
- landing_reward + progress_reward

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
