---
version: v0.2
knowledge_type: workflow
tags:
  - reward_design_agent
  - workflow
  - expert_process
  - iterative_design
should_always_include: true
used_in_stage:
  - all
---

# 00_奖励函数设计 Agent 工作流

## 1. 目标

本文件规定 LLM/Agent 在设计强化学习奖励函数时必须遵守的流程。

核心目标是让 LLM 不再直接从任务描述和 step 函数跳到 reward 代码，而是模拟专家的工作方式：

```text
先理解环境
再选择奖励骨架
再写奖励设计方案
再生成代码
再通过训练反馈迭代
```

## 2. 总体工作流

```text
Step 0：接收输入
Step 1：环境理解
Step 2：知识调用
Step 3：奖励复杂度判断
Step 4：奖励设计方案生成
Step 5：奖励黑客预检查
Step 6：奖励代码生成
Step 7：训练与评估
Step 8：失败模式诊断
Step 9：奖励函数修改
Step 10：实验记忆写入
```

---

## 3. Step 0：接收输入

Agent 应接收以下信息：

```yaml
输入:
  task_description: 任务自然语言描述
  observation_space: 观测空间定义
  action_space: 动作空间定义
  step_function: step 函数源码
  reset_function: reset 函数源码，可选
  termination_condition: 终止条件
  truncation_condition: 截断条件
  info_fields: info 字段
  existing_reward: 已有 reward，可选
  training_script: 训练脚本，可选
```

如果缺少 observation 含义或 action 含义，Agent 应尝试从源码、变量名、注释、环境文档中推断，并在输出中标注“不确定”。

---

## 4. Step 1：环境理解

Agent 必须先输出“环境理解卡片”，不能直接写 reward。

环境理解卡片至少包含：

```yaml
环境理解卡片:
  环境名称:
  任务类型:
  主目标:
  成功条件:
  失败条件:
  observation 含义:
  action 含义:
  可用于奖励的变量:
  不建议用于奖励的变量:
  可能需要的奖励骨架:
  潜在奖励黑客:
  初始奖励复杂度建议:
```

此步骤的作用是把环境源码转化为奖励设计语言。

---

## 5. Step 2：知识调用

Agent 根据环境理解卡片，从知识库中调用相关知识。

推荐使用混合调用方式：

```text
规则路由 + 标签检索 + 相似度检索
```

不要只依赖普通 embedding 相似度。

### 5.1 规则路由示例

```yaml
如果 task_type 包含 navigation:
  调用:
    - 03_任务类型到奖励骨架的决策树.md: 导航/到达目标类任务
    - 02_奖励函数骨架库.md: terminal_success, time_penalty, distance_reward, progress_delta, collision_penalty
    - 04_奖励函数失败模式库.md: goal_near_oscillation, reward_high_success_low
```

```yaml
如果 task_type 包含 locomotion 或 continuous_control:
  调用:
    - 03_任务类型到奖励骨架的决策树.md: 机器人行走/运动控制类任务
    - 02_奖励函数骨架库.md: forward_progress, alive_bonus, fall_penalty, energy_penalty, action_smoothness
    - 04_奖励函数失败模式库.md: agent_moves_fast_but_falls, agent_stands_still, jerky_action
```

```yaml
如果 reward_signal_sparse 为 true:
  调用:
    - 02_奖励函数骨架库.md: distance_reward, progress_delta, potential_based_shaping, intrinsic_reward
    - 03_任务类型到奖励骨架的决策树.md: 稀疏探索类任务
```

```yaml
如果 safety_critical 为 true:
  调用:
    - 02_奖励函数骨架库.md: constraint_penalty, gated_reward, lexicographic_reward
    - 05_奖励黑客检查清单.md
```

---

## 6. Step 3：奖励复杂度判断

Agent 必须给出奖励复杂度等级。

```text
Level 1：单一奖励
Level 2：终点奖励 + 时间/存活项
Level 3：终点奖励 + 过程引导
Level 4：目标 + 过程 + 安全 + 动作正则
Level 5：复杂复合奖励，包括塑形、探索、动态权重、多目标结构
```

选择原则：

```text
简单任务优先简单奖励。
复杂任务才使用组合奖励。
不要一开始就堆很多奖励项。
每增加一个奖励项，都必须说明作用和风险。
```

---

## 7. Step 4：奖励设计方案生成

写代码前必须输出奖励设计方案。

模板参考：

```yaml
奖励函数设计方案:
  reward_version:
  奖励复杂度等级:
  选择的奖励骨架:
  暂不选择的奖励骨架:
  奖励公式:
  每个奖励项的作用:
  每个奖励项需要的环境变量:
  预期行为:
  潜在风险:
  训练后重点观察指标:
```

如果 Agent 没有输出设计方案，不允许进入代码生成阶段。

---

## 8. Step 5：奖励黑客预检查

代码生成前，Agent 必须检查该 reward 可能被怎样钻空子。

最低检查：

```text
1. 是否能原地不动刷高分？
2. 是否能目标附近震荡刷高分？
3. 是否能猛冲拿短期奖励后失败？
4. 是否辅助奖励压过主任务？
5. 是否动作惩罚导致不敢动？
6. 是否训练 return 提高但 success_rate 不提高？
```

输出格式：

```yaml
奖励黑客预检查:
  potential_exploits:
    - exploit_id:
      description:
      severity:
      required_test:
      mitigation:
```

---

## 9. Step 6：奖励代码生成

奖励代码必须满足：

```text
1. 每个奖励项单独命名。
2. 每个奖励项写入 reward_terms。
3. 不允许只返回总 reward。
4. 不允许使用未定义变量。
5. 不允许使用未来信息。
6. 不允许使用训练时不可获得的信息。
7. 必须说明每个奖励项对应哪个 skeleton_id。
8. 必须为后续诊断保留日志字段。
```

---

## 10. Step 7：训练与评估

训练不应只记录 mean return。

必须记录：

```yaml
训练日志:
  mean_return:
  success_rate:
  failure_rate:
  episode_length:
  termination_reason:
  action_magnitude:
  action_delta:
  reward_terms_mean:
  reward_terms_sum:
  reward_terms_max:
  reward_terms_min:
```

对强化学习而言，短训存在噪声，因此推荐三阶段评估：

```text
Sanity Check：极短训练或随机/零动作测试，排除明显坏 reward
Medium Training：中等预算，判断趋势和失败模式
Long Training：长训确认最终性能
```

不要把极短训练当作最终性能结论。

---

## 11. Step 8：失败模式诊断

训练后，Agent 必须根据日志匹配失败模式。

输出：

```yaml
失败诊断:
  observed_behavior:
  matched_failure_modes:
  evidence:
  suspected_reward_causes:
  suggested_modifications:
```

示例：

```yaml
失败诊断:
  observed_behavior: 智能体快速向前但频繁摔倒
  matched_failure_modes:
    - agent_moves_fast_but_falls
  evidence:
    - forward_reward 占比过高
    - fall_count 高
    - episode_length 短
  suspected_reward_causes:
    - 前进奖励过强
    - 摔倒惩罚不足
    - 动作平滑惩罚不足
  suggested_modifications:
    - 降低 forward 权重
    - 提高 fall penalty
    - 增加 smoothness penalty
```

---

## 12. Step 9：奖励函数修改

修改 reward 时必须说明：

```yaml
reward_revision:
  previous_version:
  new_version:
  changes:
    - changed_component:
      old_value:
      new_value:
      reason:
      expected_effect:
      risk:
  expected_behavior_change:
  required_next_test:
```

禁止无理由改权重。

---

## 13. Step 10：实验记忆写入

每次实验结束必须写入结构化 Memory。

```yaml
实验记忆:
  env_name:
  task_type:
  reward_version:
  skeletons_used:
  weights:
  training_budget:
  result:
  observed_behavior:
  failure_mode:
  diagnosis:
  lesson:
  reusable_rule:
```

Memory 的作用是让系统下一次遇到类似任务时能“回忆经验”。

---

## 14. 最小 MVP 实现建议

第一版系统不需要多个 Agent。可以用一个 RewardAgent，但强制它按上面的步骤输出。

```text
RewardAgent:
  - 环境理解
  - 知识调用
  - 奖励设计
  - 奖励代码
  - 训练诊断
  - 记忆写入
```

后期再拆成：

```text
EnvironmentUnderstandingAgent
RewardArchitectAgent
RewardCoderAgent
RewardCriticAgent
ExperimentDiagnosticAgent
MemoryManager
```

---

## 15. Agent 必须遵守的底线

```text
1. 不允许跳过环境理解。
2. 不允许跳过奖励设计方案直接写代码。
3. 不允许只看 mean return 判断 reward 好坏。
4. 不允许不记录 reward_terms。
5. 不允许不做奖励黑客检查。
6. 不允许不写实验记忆。
```
