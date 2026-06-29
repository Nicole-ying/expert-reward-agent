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