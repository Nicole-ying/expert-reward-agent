# Paper-v4 架构说明 / Architecture Notes

本文档只描述产生归档 `paper_v4` 实验的历史实现，基准提交为 `cafceeb9`。

## 闭环 / Closed loop

1. **任务理解与奖励初始化。** Environment Analyzer 读取匿名任务描述和 masked step source，生成环境卡片；检索模块补充 reward-design context；Reward Generator 生成可执行初始奖励。
2. **从头训练策略。** 每个 reward candidate 都训练一个新的 PPO policy，避免沿用前一候选的 policy state。
3. **原生目标评价。** 生成的 reward 只用于 PPO 优化；候选选择使用未修改的环境 native reward。
4. **结构化观察。** Reflection Agent 读取上一轮 reward code、native score、episode length/termination 信息、组件统计、精简环境事实和历史 reward memory。
5. **诊断与受限编辑。** Agent 先形成可证伪诊断，再选择一个主要干预目标，执行 L1 尺度调整、L2 数学结构变换或 L3 主骨架重建。
6. **验证、记忆与归档。** 新代码先通过接口/安全校验，再进入下一次训练；控制器更新 reward memory，并在 native score 刷新时更新 Best Archive。
7. **继续或停止。** 达到任务阈值或耗尽固定评估预算后停止，并返回 Best Archive，而不假设最后一个候选必然最好。

This persistent `observe → diagnose → edit → train → evaluate → update` state transition is the agentic core of CREATE. “Self-evolution” refers to the reward program and its evidence-linked lineage, not to changes in LLM weights.

## 方法—代码对应 / Method-to-code map

| 论文机制 | 历史实现 | 准确边界 |
|---|---|---|
| 初始环境理解 | `pipeline/run_01_environment_analyzer_md.py` | System Prompt + task spec + masked step source |
| 初始奖励生成 | `pipeline/run_03_direct_reward_generator.py` | environment card + expert reward context |
| 闭环控制器 | `pipeline/run_iterative_experiment.py` | 调度训练、评价、反思、校验、停止和 best retention |
| PPO 与结构化反馈 | `training/train_sb3_wrapper.py` | fresh PPO、native evaluation、component statistics |
| 诊断与 L1/L2/L3 编辑 | `pipeline/run_reflection_agent.py`、`prompts/reflection_agent_prompt.md` | 每轮一个主要干预目标 |
| 持久 reward memory | `pipeline/run_06_update_reward_memory.py` | 记录版本、分数、best、诊断、编辑、预测与结果 |
| Best Archive | `pipeline/run_iterative_experiment.py` | 保存最高 native-score reward；用于最终返回和保护，不是正常 User Prompt 字段 |
| 奖励知识检索 | `rag/`、`knowledge_base/`、`pipeline/reflection_tools.py` | 初始生成使用检索上下文；反思阶段可按需调用一个知识工具 |

## Persistent Memory 与 Best Archive 的区别

- **Persistent Memory** 是给 Reflection Agent 看的 lineage state。它压缩记录过去版本的动作、预测和结果，使下一轮能够避免重复无效修改。
- **Best Archive** 是控制器维护的安全状态。native score 刷新时，控制器复制 `best_reward.py`、反馈和 summary；实验结束返回 archive 中的奖励。
- paper-v4 的正常 reflection User Prompt **没有独立 Best Code 段**。`run_reflection_agent.py` 虽接收并读取 `best_reward_path`，但 `build_user_prompt()` 中用 `if False` 排除了 best source。Best reward 只会在重复候选保护等控制分支中参与比较/重试。
- 因此框架图不应画成“Best Archive 的代码在每轮直接输入 Diagnose & Act”；准确关系是 native outcome 更新 archive，而 archive 保护最终输出。

## Paper-v4 正常迭代中不存在的机制

以下内容属于之后的探索，未参与本论文归档实验，不能写成 paper-v4 的输入或贡献：

- Subagent investigator / Subagent 调研信号
- 独立 `Component delta` Prompt 段
- 独立 `Formula switching guide` Prompt 段
- 独立“累积迭代记录”Prompt 段

历史信息由 `# 历史记忆` 中的 reward memory 承担；组件变化可以由 Agent 对当前反馈和 memory 进行比较，但当时没有单独注入一个 delta 字段。

## 精确输入关系 / Exact input relation

```text
task spec + masked step
  → Environment Analyzer
  → environment card + retrieved reward knowledge
  → initial reward R₀
  → fresh PPO training → native evaluation + component statistics
  → Reflection Agent(previous reward, feedback, environment summary, reward memory)
  → validated reward Rₜ₊₁
  → next fresh PPO training

native improvement → Best Archive → final returned reward
```
