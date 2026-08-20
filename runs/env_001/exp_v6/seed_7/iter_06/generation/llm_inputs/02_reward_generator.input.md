# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中心附近被随机初始力弹出，需要尽量快地到达并稳定停靠在中央目标垫上，同时尽可能节省引擎推力。智能体应学会：接近目标、减小速度、保持姿态稳定、安全接触目标垫。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching
- confidence: high
- reason: 虽然任务要求兼顾快速与节能，但根本目标是到达并稳定停靠在目标垫（goal reaching）。路由规则明确指出“到达某个目标状态或目标区域，通常属于 navigation_goal_reaching，即使还要求安全、稳定或高效”。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（默认）
- 各维含义：
  - obs[0] (x_position): 相对于目标垫的水平坐标
  - obs[1] (y_position): 相对于垫子高度的垂直坐标
  - obs[2] (x_velocity): 水平线速度
  - obs[3] (y_velocity): 垂直线速度
  - obs[4] (body_angle): 机体方向角
  - obs[5] (angular_velocity): 角速度
  - obs[6] (left_support_contact): 左支撑腿接触标志（0.0 或 1.0）
  - obs[7] (right_support_contact): 右支撑腿接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- 动作含义：
  - action 0: no_engine — 不点火，无推力
  - action 1: left_orientation_engine — 点燃左侧姿态引擎
  - action 2: main_engine — 点燃主引擎（下方推力）
  - action 3: right_orientation_engine — 点燃右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（当飞行器速度极低且不再移动时终止，若此时已接触目标垫则可视为成功着陆）
- failure-like termination: crash_or_body_contact（机体坠毁或非正常接触地面），horizontal_position_outside_viewport（飞出水平边界）
- ambiguous termination: body_not_awake_or_settled 本身可能发生在未接触目标垫的任何位置，需结合观测判断
- truncation: 无（返回的 truncated 永远为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，无任何字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，不能假设包含 “success”、“failure” 或 “termination_reason”

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察）
- action（当前执行的动作）
- next_obs（下一时刻观察）
- info 中明确允许的字段（本环境 info 为空，故实际不可用）

禁止使用：
- original_reward / official_reward
- training_progress（当前 prompt 未明确允许使用时必须禁止）
- 任何未声明的 info 字段
- 任何未在任务说明中出现的 obs 切片或隐藏内部状态

## 7. 可用于奖励函数的信号
- x_position
  - source: obs[0] 或 next_obs[0]
  - meaning: 水平坐标相对于目标垫
  - availability: available
- y_position
  - source: obs[1] 或 next_obs[1]
  - meaning: 垂直坐标相对于垫子高度
  - availability: available
- x_velocity
  - source: obs[2] 或 next_obs[2]
  - meaning: 水平线速度
  - availability: available
- y_velocity
  - source: obs[3] 或 next_obs[3]
  - meaning: 垂直线速度
  - availability: available
- body_angle
  - source: obs[4] 或 next_obs[4]
  - meaning: 机体方向角
  - availability: available
- angular_velocity
  - source: obs[5] 或 next_obs[5]
  - meaning: 角速度
  - availability: available
- left_support_contact
  - source: obs[6] 或 next_obs[6]
  - meaning: 左支撑腿是否接触表面（0/1）
  - availability: available
- right_support_contact
  - source: obs[7] 或 next_obs[7]
  - meaning: 右支撑腿是否接触表面（0/1）
  - availability: available
- action
  - source: action
  - meaning: 当前执行的引擎选择（可用于惩罚推力使用）
  - availability: available

## 8. 不确定或不可用的信号
- original_reward：被屏蔽，禁止使用
- info 内任何字段：info 字典为空，无可用信号
- 是否成功着陆的显式标志：不存在，必须从位置、速度、接触等多维信号自行推断
- 是否坠毁/越界的显式标志：不存在，终止条件混杂，需结合 next_obs 判断



# expert_reward_context.md

# 专家奖励知识上下文

本上下文按环境分析得到的任务类型检索。推荐骨架是候选集合，不是固定答案，也不要求全部使用。
Reward Generator 必须先检查环境卡中的变量可用性，再独立决定选择、组合、推迟或拒绝骨架。

## 任务路由

- selected_route_id: navigation_goal_reaching
- route_recommended_candidates: terminal_success_reward, terminal_failure_penalty, time_penalty, distance_reward, progress_delta_reward, potential_based_shaping, gated_reward

## 路由候选骨架

## Skeleton 1：终点成功奖励

```yaml
skeleton_id: terminal_success_reward
中文名称: 终点成功奖励
功能角色:
  - 任务目标奖励
数学形态: r_success = R_success * I[success]
适用场景:
  - 到达目标
  - 完成任务
  - 抓取成功
  - 游戏胜利
需要变量:
  - success flag
风险:
  - 奖励稀疏
  - 随机探索难以触发
常见失败模式:
  - sparse_reward_no_learning
修正:
  - 加过程引导奖励
  - 加阶段性事件奖励
  - 加势能塑形
调用规则:
  - 如果任务有明确成功条件，优先加入
```

---

---

## Skeleton 2：失败惩罚

```yaml
skeleton_id: terminal_failure_penalty
中文名称: 失败惩罚
功能角色:
  - 安全约束惩罚
数学形态: r_failure = -R_failure * I[failure]
适用场景:
  - 摔倒
  - 碰撞
  - 越界
  - 死亡
  - 任务失败
需要变量:
  - failure flag
  - termination reason
风险:
  - 太小则智能体不怕失败
  - 太大则智能体过度保守
修正:
  - 区分轻微违规和严重失败
  - 使用连续惩罚或门控奖励
调用规则:
  - 如果环境有 failure/terminated 条件，必须考虑
```

---

---

## Skeleton 3：每步时间惩罚

```yaml
skeleton_id: time_penalty
中文名称: 每步时间惩罚
功能角色:
  - 行为效率约束
数学形态: r_time = -lambda_time
适用场景:
  - 希望尽快完成任务
  - 导航
  - 路径规划
  - 迷宫
风险:
  - 太大可能导致冒险行为
  - 对需要等待的任务可能有害
修正:
  - 使用较小权重
  - 与成功奖励配合
调用规则:
  - 如果目标是尽快完成任务，加入
```

---

---

## Skeleton 5：距离型密集奖励

```yaml
skeleton_id: distance_reward
中文名称: 距离型密集奖励
功能角色:
  - 过程引导奖励
数学形态: r_distance = -d(s_t, goal)
适用场景:
  - 目标位置明确
  - 可以定义距离
  - 成功奖励稀疏
需要变量:
  - 当前状态位置
  - 目标位置
风险:
  - 接近目标但不完成任务
  - 距离函数不等于真实任务进度
修正:
  - 加终点成功奖励
  - 加时间惩罚
  - 改成增量进步奖励
调用规则:
  - 如果可定义距离且稀疏奖励难学，可加入
```

---

---

## Skeleton 6：增量进步奖励

```yaml
skeleton_id: progress_delta_reward
中文名称: 增量进步奖励
功能角色:
  - 过程引导奖励
数学形态: r_progress = d(s_t, goal) - d(s_{t+1}, goal)
适用场景:
  - 导航
  - 接近目标
  - 机器人移动
  - 机械臂接近物体
需要变量:
  - 上一步距离
  - 当前距离
风险:
  - 目标附近震荡刷分
  - 只优化局部进展
常见失败模式:
  - goal_near_oscillation
修正:
  - 加终点奖励
  - 加时间惩罚
  - 裁剪 progress
调用规则:
  - 如果任务有“越来越接近目标”的结构，优先考虑
```

---

---

## Skeleton 14：势能塑形奖励

```yaml
skeleton_id: potential_based_shaping
中文名称: 势能塑形奖励
功能角色:
  - 过程引导奖励
  - 奖励塑形
数学形态:
  F(s,a,s') = gamma * Phi(s') - Phi(s)
  R' = R + F
适用场景:
  - 稀疏奖励
  - 可以定义状态好坏函数 Phi
风险:
  - Phi 设计错误会误导学习
  - 塑形项压过原始任务奖励
修正:
  - 让 Phi 与真实目标一致
  - 记录 shaping term
文献依据:
  - Ng et al. 1999 potential-based reward shaping
调用规则:
  - 如果希望提供学习引导且尽量保持原目标，考虑
```

---

---

## Skeleton 10：门控/层级奖励

```yaml
skeleton_id: gated_reward
中文名称: 门控/层级奖励
功能角色:
  - 安全优先结构
数学形态:
  if unsafe:
      r = -R_large
  else:
      r = r_task
适用场景:
  - 安全优先
  - 自动驾驶
  - 医疗控制
  - 碰撞不可接受
风险:
  - 太保守
  - unsafe 判断过宽导致学习困难
修正:
  - 区分硬约束和软约束
  - 硬约束门控，软约束连续惩罚
调用规则:
  - 如果某个约束不能被其他奖励抵消，使用
```

---

---

## 其他可用骨架目录

terminal_success_reward, terminal_failure_penalty, time_penalty, alive_bonus, distance_reward, progress_delta_reward, forward_progress_reward, event_reward, weighted_sum_reward, gated_reward, energy_penalty, action_smoothness_penalty, stability_penalty, potential_based_shaping, intrinsic_exploration_reward, dynamic_curriculum_reward, learned_preference_reward

## 选择规则

- 不机械照抄路由推荐，也不默认任何骨架是主信号。
- 只选择环境卡中有可靠输入支持的骨架。
- 根据任务目标判断是否需要任务目标、过程引导、约束、效率、探索或课程角色。
- 最小设计意味着删除无证据的项，而不是固定组件数量或固定组件组合。
- 没有显式 success/failure 信号时，不得发明对应字段。
- 需要其他骨架时，可以从完整目录中选择，但必须说明接口依据和风险。
- 中间量和门控系数不是独立 reward term，不应放入 components 与奖励贡献比较。



# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- angle_penalty + angular_vel_penalty + distance_reward + energy_penalty + time_penalty
- landing_bonus + progress_delta + stability_penalty

请选择一个**完全不同的主信号骨架**。例如如果上述列表都是 progress_delta 系列，请尝试 potential_based_shaping 或 bounded_proximity。不要重复已失败的骨架。
