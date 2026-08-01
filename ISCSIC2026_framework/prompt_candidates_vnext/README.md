# Initial-stage Prompt redesign (vNext candidate)

本目录是对环境理解与初始奖励生成 Prompt 的候选重构，不属于历史 `paper_v4` 执行快照，也不会替换 `../prompts/` 中用于来源审计的原始 Prompt。

设计目标：用最短的信息链回答四个问题——任务怎样才算成功、怎样会失败、哪些信号合法、什么奖励方向会把最优策略引向任务成功。

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
