# 10_v7_reflection_rag 说明

v7 先实现三个目标：

```text
1. 修复 02 的 17 个骨架精确切片漏项；
2. 把 02/03/04/05 的表格单独抽成 table chunks；
3. 增加 reflection RAG：训练后只查 04/05。
```

## 1. 02 的 17 个骨架

v6 中 `forward_progress_reward` 和 `weighted_sum_reward` 因为标题不完全匹配而漏切。  
v7 改成按 `Skeleton 编号` 匹配，而不是按完整中文标题匹配。

现在应完整切出：

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

## 2. 表格 chunks

v7 新增：

```text
knowledge_base/stage_generation/generation_table_chunks_02_03.jsonl
knowledge_base/stage_reflection/reflection_table_chunks_04_05.jsonl
```

表格不再混在正文里，而是作为 table_context 单独进入 RAG 输出。

### 生成阶段表格

来自：

```text
02_奖励函数骨架库.md
03_任务类型到奖励骨架的决策树.md
```

用于辅助任务路由与骨架选择。

### 反思阶段表格

来自：

```text
04_奖励函数失败模式库.md
05_奖励黑客检查清单.md
```

用于训练后快速诊断与奖励黑客检查。

## 3. Reflection RAG

新增：

```text
pipeline/run_06_reflection_retrieval.py
```

输入：

```text
training_summary.json 或 --mock 生成的 mock_training_summary.json
reward_design_plan.json
```

输出：

```text
retrieved_reflection_knowledge.json
retrieved_reflection_knowledge.md
```

只允许检索：

```text
04_奖励函数失败模式库.md
05_奖励黑客检查清单.md
```

禁止检索：

```text
00/01/02/03
```

## 4. 当前还没做的

v7 没有做完整 diagnosis LLM、memory update、tools。  
这三个属于下一步。
