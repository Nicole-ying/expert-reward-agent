# 08_v5_stage_rag 说明

## 核心修改

v5 不再让 00/01 参与 RAG 检索。00/01 的作用变为固定 prompt 原则。

## 阶段分工

奖励生成阶段只检索：

```text
02_奖励函数骨架库.md
03_任务类型到奖励骨架的决策树.md
```

反思/诊断阶段只检索：

```text
04_奖励函数失败模式库.md
05_奖励黑客检查清单.md
```

## controlled_task_tags 的替代

v5 不再用 controlled_task_tags 直接触发检索，而是：

```text
task_type_analysis：用于检索 03 任务路由
reward_skeleton_screening：从 02 的 17 个骨架里筛选 candidate/defer/reject
retrieval_request：明确告诉 RAG 检索哪些 skeleton_id 和 route query
```
