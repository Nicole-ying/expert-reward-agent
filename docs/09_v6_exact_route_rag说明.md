# 09_v6_exact_route_rag 说明

## 1. v6 修正的问题

v5 虽然把生成阶段限制在 02/03，但检索仍然会因为关键词相似度拿到多个 03 路由，例如机械臂、自动驾驶等不相关任务路由。

v6 改为严格的两步精确检索：

```text
Environment Analyzer：
  从 03 的 7 个 route_id 中选择且只选择 1 个 selected_route_id
  从 02 的 17 个 skeleton_id 中筛 candidate / defer / reject

Generation RAG：
  精确检索 selected_route_id 对应的 03 章节
  精确检索 candidate/deferred/route_recommended 对应的 02 骨架章节
```

## 2. 参与切片和匹配的文件

### 生成阶段 active RAG

```text
02_奖励函数骨架库.md
03_任务类型到奖励骨架的决策树.md
```

### 反思阶段 active RAG

```text
04_奖励函数失败模式库.md
05_奖励黑客检查清单.md
```

### 不参与生成阶段 RAG

```text
00_奖励函数设计Agent工作流.md
01_奖励函数设计总原则.md
06_奖励函数代码模板.md
07_环境理解卡片模板.md
08_奖励函数设计方案模板.md
09_实验记忆写入模板.md
10_文献依据与参考资料.md
```

## 3. 7 个任务类型

```text
survival_balance_task
navigation_goal_reaching
locomotion_continuous_control
manipulation_grasping
autonomous_driving_safety
sparse_exploration
multi_objective_task
```

## 4. 17 个奖励骨架

```text
terminal_success_reward
terminal_failure_penalty
time_penalty
alive_bonus
distance_reward
progress_delta_reward
forward_progress_reward
event_reward
weighted_sum_reward
gated_reward
energy_penalty
action_smoothness_penalty
stability_penalty
potential_based_shaping
intrinsic_exploration_reward
dynamic_curriculum_reward
learned_preference_reward
```
