# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标
这是一个2D飞行器轨迹优化任务。一个飞行器从视口顶部附近开始，受到初始随机力的作用。目标是尽可能快地到达并稳定在中央目标着陆平台上，同时尽可能少地使用引擎推力。智能体需要学会接近目标、降低速度、保持稳定姿态并安全接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务明确要求"到达并稳定在中央目标着陆平台"，核心目标是导航到目标位置并稳定停靠，同时优化燃料消耗。这符合导航目标到达任务的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position - 相对于目标着陆平台的水平坐标
- obs[1]: y_position - 相对于着陆平台高度的垂直坐标
- obs[2]: x_velocity - 水平线速度
- obs[3]: y_velocity - 垂直线速度
- obs[4]: body_angle - 飞行器姿态角度
- obs[5]: angular_velocity - 角速度
- obs[6]: left_support_contact - 左侧支撑接触标志 (0.0 或 1.0)
- obs[7]: right_support_contact - 右侧支撑接触标志 (0.0 或 1.0)

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine - 不执行任何操作
- action 1: left_orientation_engine - 启动左侧姿态引擎
- action 2: main_engine - 启动主引擎
- action 3: right_orientation_engine - 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled - 当飞行器停止运动并稳定在着陆平台上时触发，可能是成功终止
- failure-like termination: crash_or_body_contact - 坠毁或非正常机体接触；horizontal_position_outside_viewport - 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 无显式截断，但可能由环境内部处理

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false - 没有明确的成功标志
- explicit_failure_flag_available: false - 没有明确的失败标志
- allowed_info_fields: 无（step返回空字典{}）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs - 当前状态观测
- action - 当前执行的动作
- next_obs - 执行动作后的下一状态观测
- info - 当前为空字典，无可用字段
- training_progress - 仅当prompt明确允许时才使用

禁止使用：
- original_reward - 官方奖励已被屏蔽，禁止使用
- official_reward - 禁止使用
- 未声明的info字段 - info为空字典
- 未声明的obs切片 - 仅允许使用obs[0]到obs[7]的已声明字段

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) - 相对于目标的位置
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) - 线速度
- orientation: obs[4] (body_angle), obs[5] (angular_velocity) - 姿态和角速度
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) - 支撑接触标志
- action/engine: action (0-3) - 引擎使用情况，可用于惩罚燃料消耗

## 8. 不确定或不可用的信号
- 目标位置绝对坐标：不可用，只有相对位置
- 着陆平台高度：不可用，只有相对y坐标
- 初始随机力：不可用，无法从观测中获取
- 燃料剩余量：不可用，未在观测空间中提供
- 时间步数：不可用，未在观测或info中提供
- 距离目标的绝对距离：不可用，需要从obs[0]和obs[1]计算
- 成功/失败标志：不可用，info为空字典
- 终止原因：不可用，无法区分是成功还是失败终止



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

## 1. 检索策略
- selected_route_id: navigation_goal_reaching
- 生成阶段只使用 02 奖励骨架库 与 03 任务类型路由知识。
- 这里是给 Reward Generator 直接阅读的压缩专家上下文，不是完整知识库。

## 2. 当前任务类型专家知识
### 4. 导航 / 到达目标类任务
## 4. 导航 / 到达目标类任务

```yaml
route_id: navigation_goal_reaching
中文名称: 导航/到达目标类任务
触发条件:
  - objective 包含 reach_goal / navigation / go_to_target / move_to_position
  - observation 或 info 中存在 goal_position / distance_to_goal / x,y position
典型环境:
  - 迷宫
  - 移动机器人导航
  - 小车到达目标点
推荐奖励骨架:
  必选:
    - terminal_success_reward
    - terminal_failure_penalty
    - time_penalty
  稀疏时加入:
    - distance_reward
    - progress_delta_reward
    - potential_based_shaping
  有障碍时加入:
    - collision_penalty
    - gated_reward
重点失败模式:
  - goal_near_oscillation
  - reward_high_success_low
  - event_reward_farming
必须检查:
  - goal_oscillation_exploit
  - high_reward_without_success
初始复杂度建议: Level 3
```

推荐初始公式：

```math
r_t = R_{success} I[goal] - R_{fail} I[collision] - \lambda_{time} + \lambda_p(d_t - d_{t+1})
```

专家解释：

```text
导航任务通常不能只用终点奖励，因为太稀疏。
增量奖励能告诉智能体“这一步是否更接近目标”。
但增量奖励容易导致目标附近震荡，所以必须配合成功奖励和时间惩罚。
```

---

推荐骨架：
- terminal_success_reward
- terminal_failure_penalty
- time_penalty
- distance_reward
- progress_delta_reward
- potential_based_shaping
- gated_reward

## 3. 相关奖励骨架知识
### terminal_success_reward
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

### terminal_failure_penalty
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

### time_penalty
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

### distance_reward
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

### progress_delta_reward
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

### gated_reward
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

### energy_penalty
## Skeleton 11：动作能耗惩罚

```yaml
skeleton_id: energy_penalty
中文名称: 动作能耗惩罚
功能角色:
  - 行为正则惩罚
数学形态: c_energy = lambda_a * ||a_t||^2
适用场景:
  - 连续动作空间
  - 机器人
  - 机械臂
  - 无人车
需要变量:
  - action
风险:
  - 太大导致不动
常见失败模式:
  - agent_afraid_to_move
修正:
  - 降低权重
  - 对动作范围归一化
调用规则:
  - 如果 action 是连续控制量，考虑加入
```

---

### stability_penalty
## Skeleton 13：姿态/稳定性惩罚

```yaml
skeleton_id: stability_penalty
中文名称: 姿态/稳定性惩罚
功能角色:
  - 安全约束惩罚
  - 行为正则惩罚
数学形态: c_stability = lambda_theta * ||theta_t||^2
适用场景:
  - 机器人行走
  - 倒立摆
  - 无人机控制
需要变量:
  - 角度
  - 角速度
风险:
  - 太大导致过度保守
修正:
  - 与进度奖励平衡
  - 使用合理角度范围裁剪
调用规则:
  - 如果姿态稳定性决定任务成功，加入
```

---

### potential_based_shaping
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

## 4. 本阶段生成要求
- 直接生成 reward_v1.py。
- 从简单到复杂，但不要把 minimal-first 理解成只能用一个组件。
- 如果 success/failure 显式信号不存在，不要用 terminal_success_reward / terminal_failure_penalty 作为 v1 核心项。
- 优先使用明确可由 obs/next_obs/action 得到的信号。
- 对目标到达且存在速度/姿态风险的任务，建议考虑：主学习信号 + 轻量稳定约束。
- 后续迭代再考虑 terminal、failure、energy、time、gated 等项。