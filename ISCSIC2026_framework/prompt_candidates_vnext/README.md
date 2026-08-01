# Initial-stage Prompt redesign (vNext candidate)

本目录是对环境理解与初始奖励生成 Prompt 的候选重构，不属于历史 `paper_v4` 执行快照，也不会替换 `../prompts/` 中用于来源审计的原始 Prompt。

设计目标：保留原始任务描述，用最短的信息链回答任务类型、空间索引、怎样才算成功、怎样会失败、哪些信号合法，以及什么奖励方向会把最优策略引向任务成功。

- `DIAGNOSIS.md`：现有 Prompt 的问题与改写原则。
- `01_environment_semantics_prompt.md`：精简环境语义分析 System Prompt。
- `02_initial_reward_generator_prompt.md`：少组件、目标主导的初始奖励生成 System Prompt。

建议的新 User Prompt 数据流：

```text
task spec + masked step source
  → concise environment semantics card
  → initial reward generator
  → 2–4 component reward
```

默认不再向初始生成器注入大段通用专家骨架目录。只有环境语义存在无法从源码解决的歧义时，才按需检索一条针对性知识，而不是整包注入。

Initial Reward Generator 对组件数量使用硬约束：`components` 必须恰好包含 2–4 个实际进入 `total_reward` 的具名项；超过四项的职责留给后续训练证据驱动的修复。

最终 Environment Card 应由控制器与 LLM 共同组装：控制器把匿名任务描述原样写入第 0 节，避免 LLM 二次改写；Environment Analyzer 生成第 1–8 节。第 0–5 节是共享任务接口，包括任务原文、最小任务类型、observation 表、action 表、终止/截断和合法信号，Reflection Agent 每轮只读取这些部分。第 6–8 节包含主进展骨架、失败模式和初始 reward brief，只提供给 Initial Reward Generator，避免反思阶段重复注入初始化建议。

建议的组装形式：

```text
# Environment Semantics Card

## 0. Original anonymized task specification
{controller inserts task description verbatim}

{Environment Analyzer output: sections 1–8}
```
