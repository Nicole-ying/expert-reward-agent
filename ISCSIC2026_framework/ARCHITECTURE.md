# Paper-v4 architecture / Paper-v4 论文框架说明

This document maps the paper's method description to the submitted implementation.

中文解释：本文档用于说明论文里的每一个方法模块在代码中如何实现，以及各模块之间的数据关系。英文名称与论文保持一致，中文部分帮助理解，不影响代码运行。

## Closed loop / 闭环流程

1. **Task interface and initialization.** The environment card, task specification, masked transition interface, and retrieved reward-design evidence ground an initial executable reward `R_0`.

   **中文解释：任务接口与奖励初始化。** 系统首先读取任务描述、匿名化环境接口和检索到的奖励设计知识，在不接触官方奖励实现的条件下生成第一个可执行奖励 `R_0`。

2. **Fresh policy training and native evaluation.** Every candidate reward trains a fresh PPO policy. Selection uses the unchanged environment reward, so CREATE cannot grade itself with its generated shaping reward.

   **中文解释：重新训练策略并使用原生目标评估。** 每一个候选奖励都从头训练一个新的 PPO 策略。生成的奖励只负责训练，候选奖励的好坏由环境没有被修改的原生任务分数判断，因此 CREATE 不能用自己编写的塑形奖励给自己打分。

3. **Observe.** The agent receives the native score, training trace, per-term reward statistics (`episode_sum_mean`, `signed_share`, `magnitude_share`, and `active_rate`), the active reward, and its lineage record.

   **中文解释：观察。** Agent 不只读取一个最终分数，还会读取训练曲线、当前奖励代码、历史修改记录，以及每个奖励分量的平均贡献、正负方向、幅值占比和激活率。这些信息用于判断奖励的哪一部分出了问题。

4. **Diagnose and act.** The reward editor forms one diagnosis and chooses one primary target. It applies an L1 parameter tune, L2 structural refactor, or L3 reward redesign.

   **中文解释：诊断并采取动作。** Agent 先形成一个可以被下一轮训练验证的故障假设，再选择一个主要修改目标。L1 表示调整权重或阈值；L2 表示改变某个奖励分量的数学结构；L3 表示在局部修改多次失败后重新设计奖励分解。

5. **Validate, remember, and archive.** The candidate is checked for interface and safety violations. The transition record is appended to persistent memory; the best archive changes only when native evaluation improves.

   **中文解释：验证、记忆与归档。** 新奖励在训练前必须通过接口、语法和安全检查。本轮的“证据—诊断—修改—结果”被写入 Persistent Memory；只有原生任务表现确实提高时，Best Archive 才会被更新。Persistent Memory 保存整个推理谱系，Best Archive 只保存当前最优解，两者不能混为一谈。

6. **Repeat or stop.** The validated reward becomes `R_{t+1}`. Search stops at the task threshold or the fixed evaluation budget.

   **中文解释：继续迭代或停止。** 通过验证的奖励成为下一轮的 `R_{t+1}`。一旦原生任务分数首次达到阈值，或者耗尽最多 10 次奖励评估预算，当前搜索谱系停止。

This stateful observe-diagnose-act-update cycle is the reason CREATE is an agent rather than a sequence of unrelated LLM calls.

中文解释：CREATE 被称为 Agent 的关键并不是“调用了 LLM”，而是它拥有跨轮次持续存在的状态，并形成“观察—诊断—行动—环境验证—状态更新”的完整闭环。每一次策略训练相当于 Agent 与外部环境之间的一次昂贵状态转移。

## Method-to-code map / 方法与代码对应关系

| 论文机制 | 主要实现位置 | 中文说明 |
|---|---|---|
| Iterative closed loop | `pipeline/run_iterative_experiment.py` | 控制奖励生成、训练、评估、修复和停止条件的主循环 |
| Structured training evidence | `training/train_sb3_wrapper.py` | 训练 PPO，并统计奖励分量激活率、幅值占比和原生评估结果 |
| Diagnosis and bounded L1/L2/L3 edit | `pipeline/run_reflection_agent.py`、`prompts/reflection_agent_prompt.md` | 根据证据形成诊断，并约束 Agent 只执行一次明确的 L1、L2 或 L3 修改 |
| Optional investigator signal | `pipeline/subagent_investigator.py` | 对复杂训练现象提供补充分析信号；最终修改仍由主奖励编辑 Agent 决定 |
| Reward validation and repair guard | `pipeline/run_iterative_experiment.py` 中的验证逻辑 | 在消耗 PPO 训练预算前检查奖励是否可执行、是否违反接口约束 |
| Persistent lineage memory | `pipeline/run_06_update_reward_memory.py` 和 `runs/` 中的实验记录 | 保存历次奖励、诊断、修改、预测和真实结果 |
| Best-reward archive | `pipeline/run_iterative_experiment.py` 中的 best-retention 逻辑 | 始终保留原生任务得分最高的奖励，防止后续失败覆盖成功结果 |
| Expert reward context | `rag/`、`knowledge_base/`、`pipeline/run_02_build_expert_context.py` | 为初始奖励设计和修复提供检索到的奖励设计知识 |
| Fresh PPO and native evaluation | `training/train_sb3_wrapper.py` | 每次候选奖励重新训练策略，并使用未修改的环境奖励进行外部评价 |

## Paper-aligned configurations / 与论文对应的实验配置

- `configs/env001_paper_v4.yaml`

  中文解释：论文中 CREATE 的完整实验条件。包含 5 条独立谱系、结构化训练证据、Persistent Memory、受限编辑和辅助调查信号。

- `configs/env001_ablation_score_only_v4.yaml`

  中文解释：仅分数反馈消融。Agent 看不到奖励分量的激活率和幅值占比，用于验证单个任务分数是否足以指导修复。

- `configs/env001_ablation_eureka_feedback_v4.yaml`

  中文解释：粗粒度反馈消融。只提供任务分数和奖励分量均值，不提供 CREATE 使用的完整结构化诊断表。

- `configs/env001_ablation_unconstrained_v4.yaml`

  中文解释：无约束编辑消融。保留结构化证据和记忆，但取消单目标限制以及 L1/L2/L3 编辑合同。这个条件测试的是整套受限编辑机制，不能分别证明其中某一个子机制的独立贡献。

- `scripts/run_independent_baseline.sh`

  中文解释：同提示、同预算的独立奖励生成 baseline。它使用相同任务上下文、LLM 配置和 PPO 成本，但每次重新生成无关候选，不读取上一轮证据，也没有持续记忆与修复。

- `scripts/run_official_baseline.sh`

  中文解释：直接使用环境官方奖励训练 PPO 的参考条件，奖励适配器位于 `baselines/official_reward.py`。它用于说明任务的标准训练表现，不是 LLM 奖励搜索 baseline。

## Semantic boundaries / 论文中的概念边界

- **Training reward / 训练奖励：** 可编辑的奖励程序，PPO 实际优化的信号。
- **Native outcome / 原生任务结果：** 环境未修改的任务得分，只用于选择奖励和报告结果。
- **Persistent Memory / 持久化记忆：** 按时间记录奖励、诊断、修改和结果的完整谱系，不等同于最优档案。
- **Best Archive / 最优档案：** 当前原生任务结果最好的奖励与策略；最新候选不会自动覆盖它。
- **Bounded repair / 受限修复：** 每轮围绕一个主要故障目标执行一次声明清楚的 L1、L2 或 L3 修改，并在训练前验证代码。
- **Self-evolution / 自进化：** 奖励程序和诊断谱系在外部训练反馈下持续演化；LLM 权重保持不变。因此论文主张的是奖励程序层面的自进化，而不是基础模型自主学习。
