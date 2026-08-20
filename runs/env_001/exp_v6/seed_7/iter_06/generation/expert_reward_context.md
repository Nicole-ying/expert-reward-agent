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