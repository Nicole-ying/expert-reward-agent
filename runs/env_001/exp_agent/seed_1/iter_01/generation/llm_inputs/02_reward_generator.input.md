# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
本环境是一个 2D 车辆式轨迹优化任务。智能体（一个主体）从视口顶部中心附近开始，受到一个随机的初始作用力。  
目标是在最短时间内飞到并稳定停靠在视口中心的**目标垫**上，同时**尽量少使用引擎推力**。  
智能体需要学会：
- 接近目标垫中心；
- 降低水平与垂直速度；
- 保持姿态稳定（机身角度接近水平）；
- 安全接触（用脚而不是机身其他部位）。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: medium  
reason: 任务包含多个优化目标（快速到达、低燃料消耗、姿态稳定、安全着陆），且这些目标之间存在权衡（例如快速到达与低燃料消耗矛盾），属于典型的多目标任务。也可理解为“导航目标到达”，但着陆姿态和燃料约束使多目标特性更突出。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（实际值连续，部分字段为 0/1 布尔值转换为浮点）
- **obs[0]**: `x_position` —— 主体相对于目标垫中心的水平坐标（单位未给出，通常归一化或像素坐标）
- **obs[1]**: `y_position` —— 主体相对于目标垫**高度**的垂直坐标（从垫子平面往上测量）
- **obs[2]**: `x_velocity` —— 水平线速度
- **obs[3]**: `y_velocity` —— 垂直线速度
- **obs[4]**: `body_angle` —— 机身角度（方向，0 为理想水平）
- **obs[5]**: `angular_velocity` —— 角速度
- **obs[6]**: `left_support_contact` —— 左支撑脚是否与物体接触（1.0 表示接触，0.0 表示未接触）
- **obs[7]**: `right_support_contact` —— 右支撑脚是否与物体接触（同上）

## 4. 动作空间 action_space
- type: Discrete(4)
- **action 0**: `no_engine` —— 不点火，无任何推进
- **action 1**: `left_orientation_engine` —— 启动**左方**姿态引擎（产生旋转力矩，使机身向右旋转趋势）
- **action 2**: `main_engine` —— 启动**主引擎**（产生向上的推力，用于减速/升空）
- **action 3**: `right_orientation_engine` —— 启动**右方**姿态引擎（产生旋转力矩，使机身向左旋转趋势）

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `terminated = crash_or_body_contact OR horizontal_position_outside_viewport OR body_not_awake_or_settled`：
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当主体完全稳定（settled）或不再活跃（not awake）时触发。若此时主体位于目标垫上，则视为**成功着陆**。但需由观察中的位置进一步确认。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 主体发生碰撞（如撞到地面、墙壁）或**机身本体**与物体接触（非脚触），通常表示坠毁。  
  - `horizontal_position_outside_viewport` —— 主体水平位置超出可视区域，表示飞出边界。
- **ambiguous termination**:  
  `body_not_awake_or_settled` —— 可能出现在目标垫之外的其他位置稳定（例如停在边角地面），此时应为失败。仅靠终止条件无法区分，需结合位置观察。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** —— `step()` 返回的 `info` 为空字典，没有提供 `success` 或 `failure` 标志。
- explicit_failure_flag_available: **false** —— 同上。
- allowed_info_fields: 无（`info` 在任何时候均为空，无法从中提取任何字段）。
- forbidden_or_uncertain_info_fields: 所有 `info` 字段均不可用，包括可能假设的 `success`, `failure`, `termination_reason` 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`：当前步的状态（8维，可直接使用全部或部分维度）
- `action`：当前步采取的动作（0~3 的整数）
- `next_obs`：执行动作后的下一状态（8维）
- `info`：空的 Python 字典，不可使用任何字段
- `training_progress`：仅在明确指明允许时可用，此处任务未要求使用，**不宜依赖**

**禁止使用**：
- `original_reward`（`MASKED_OFFICIAL_REWARD`）
- 任何命名为 `official_reward` 的变量
- 未声明的 `info` 字段（`info` 为空，因此所有字段均禁止）
- 未声明的 `obs` / `next_obs` 切片含义（如硬编码的特定维度以外的维度含义）
- 假设环境内部变量（如碰撞标志、离地高度等）

## 7. 可用于奖励函数的信号
所有观察空间字段均可用，具体如下：
- **position**: `x_position`（obs[0] 和 next_obs[0]）、`y_position`（obs[1] 和 next_obs[1]）
- **velocity**: `x_velocity`（obs[2] 和 next_obs[2]）、`y_velocity`（obs[3] 和 next_obs[3]）
- **orientation**: `body_angle`（obs[4] 和 next_obs[4]）、`angular_velocity`（obs[5] 和 next_obs[5]）
- **contact**: `left_support_contact`（obs[6] 和 next_obs[6]）、`right_support_contact`（obs[7] 和 next_obs[7]）
- **action/engine**: 动作 `action` （0-3）本身，可用于燃料惩罚等

## 8. 不确定或不可用的信号
- 官方原始奖励（已被掩码）
- 任何 `info` 中的字段（因为为空）
- 终止原因的具体标签（`crash_or_body_contact`、`horizontal_position_outside_viewport`、`body_not_awake_or_settled` 均未暴露在 `info` 中）
- 目标垫的绝对尺寸、燃料剩余量、视口边界坐标等未在观察中提供的环境内部变量
- `training_progress` 不明是否可用，默认不使用



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。

## 1. 任务路由摘要
- multi_objective_task：按该任务类型选择主学习信号，并先检查接口可用性。

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