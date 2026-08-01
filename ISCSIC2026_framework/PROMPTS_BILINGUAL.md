# Paper-v4 Prompt 中英文归档 / Bilingual Prompt Archive

## 0. 版本边界 / Version boundary

本文件描述历史提交 `cafceeb9` 中真实执行的 Prompt。中文 System Prompt 原文以 `prompts/` 下三个文件为唯一可执行真值；下方英文为语义对照译文，不参与运行。

This document describes the prompts actually executed by historical commit `cafceeb9`. The three Chinese files under `prompts/` are the executable source of truth. The English text below is a semantic reference translation and is not executed.

paper-v4 只有三个 LLM 角色：

1. Environment Analyzer：读取匿名任务接口，生成 environment card。
2. Initial Reward Generator：读取 environment card 与检索上下文，生成 `reward_v1`。
3. Reflection Agent：读取上一轮奖励、训练反馈、精简环境事实和 reward memory，生成下一版奖励。

There is no Subagent investigator in paper-v4. There is also no independent Component-delta, Formula-switching-guide, or cumulative-record block in the normal reflection prompt.

---

## 1. Environment Analyzer

### 1.1 System Prompt — 中文原文

可执行全文：`prompts/01_environment_analyzer_prompt.md`

职责与约束完整摘要：

- 只负责读懂匿名环境，输出 Markdown environment card；不生成奖励代码。
- 从七类任务路线中选一个 `selected_route_id`，再给出更细的 `dynamics_subtype`。
- 明确 observation/action space、step/termination、允许和禁止使用的 reward 输入。
- 输出 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 和训练后需要检查的 failure modes。
- 不得恢复官方 reward、猜测真实环境名称、发明 `info` 字段或未声明 observation slice。
- 输出必须包含 1–12 节：任务目标、任务类型、观测、动作、终止、reward 接口、可用信号、不可用信号、任务画像、reward roles、role-signal mapping、failure modes。

### 1.2 System Prompt — English translation

You are the reinforcement-learning environment-understanding module. Your sole responsibility is to interpret an anonymized environment and produce a stable, reusable Markdown environment card for downstream reward generation and reflection.

Required behavior:

- Select exactly one coarse task family and one finer dynamics subtype.
- Describe the observation space, action space, transition/termination semantics, and the reward-function interface.
- Separate usable signals from unavailable or uncertain signals.
- Produce an expert task profile, a decomposition of mandatory/conditional/forbidden reward roles, a role-to-signal map, and post-training failure modes to inspect.
- Do not generate reward code, reproduce an official reward, reveal or guess the real benchmark identity, invent `info` fields, or assume undeclared observation slices.
- Output the prescribed twelve-section Markdown environment card.

### 1.3 User Prompt — 实际模板 / Actual template

```text
ANONYMIZED_TASK_SPEC:
{complete contents of task_spec_anonymized.yaml}

MASKED_STEP_SOURCE:
{complete contents of masked_step_source.py}
```

中文解释：User Prompt 不额外加入搜索历史或 reward code，只提供匿名任务规范与 masked step source。

English: The user message contains only the anonymized task specification and the masked transition source. It does not include search history or reward code.

---

## 2. Initial Reward Generator

### 2.1 System Prompt — 中文原文

可执行全文：`prompts/02_reward_generator_prompt.md`

职责与约束完整摘要：

- 按 `role → signal → formula operator → code` 顺序设计第一版奖励，而不是机械套用 skeleton。
- environment card 的 reward-role decomposition 高于检索模板；没有可用信号的 role 必须排除。
- v1 优先覆盖一个主学习信号和必要的健康/安全约束，通常使用 2–4 个 component；效率、强门控、curriculum 等默认后置。
- 只使用 environment card 明示的 `obs`、`next_obs`、`action`、`info`；禁止 `original_reward`、未声明字段、`fitness_score` 等。
- 第一段 Python code block 必须只含完整 `compute_reward`，返回 `(float(total_reward), components)`。
- 不允许 import、class、额外函数、`self`、`eval/exec/open`。
- 代码后说明 task family、dynamics subtype、selected/excluded roles、signal mapping、formula operators 和待观察 failure modes。

### 2.2 System Prompt — English translation

You are the initial reward-function generator. Design the first reward in the order `reward role → available signal → mathematical operator → executable code`; do not mechanically copy a named skeleton.

Required behavior:

- Treat the environment card’s role decomposition and interface constraints as authoritative; retrieved expert material is design guidance, not a fixed answer.
- Exclude any role whose required signal is unavailable.
- Start with one dense primary learning signal plus only necessary health/safety constraints. A 2–4 component v1 is recommended, while efficiency costs, strong gates, and curricula are normally deferred.
- Use only declared `obs`, `next_obs`, `action`, and allowed `info` fields. Never use the original environment reward or invent inputs.
- The first Python block must contain exactly one complete `compute_reward` function and return `(float(total_reward), components)`.
- Do not use imports, classes, helper functions, `self`, `eval`, `exec`, or file access.
- After the code, explain the chosen task profile, roles, signal mapping, mathematical operators, excluded roles, and failure modes to inspect after training.

### 2.3 User Prompt — 正常生成 / Normal generation

```text
# environment_card.md
{complete generated environment card}

# expert_reward_context.md
{retrieved expert reward-design context}
```

在 paper-v4 主配置中，`include_masked_step_in_reward_generator: false`，所以 Reward Generator 的 User Prompt 不重复注入 masked step source。

In the paper-v4 main configuration, `include_masked_step_in_reward_generator` is false, so the masked step source is not injected a second time into this user message.

若控制器触发 fresh restart，会在尾部加入 `restart_context.md`。若 v1 代码验证失败，会继续加入：

If the controller triggers a fresh restart, `restart_context.md` is appended. If v1 validation fails, the following repair block is appended:

```text
# Validation repair
具体错误：{validation error}
只修复代码合规问题，不重新分析环境，不改变原定奖励设计。直接输出完整合规的compute_reward函数。

# Invalid previous draft
{invalid draft}
```

English meaning: fix only the reported code-compliance error; do not redesign the reward or re-analyze the environment, and return a complete valid function.

---

## 3. Reflection Agent

### 3.1 System Prompt — 中文原文

可执行全文：`prompts/reflection_agent_prompt.md`

职责与约束完整摘要：

- 用训练证据解释失败，再选择最小、可验证的单一干预。
- Component statistics 是观察证据，不是因果贡献；必须与 native score、episode length、termination distribution 和历史干预共同解释。
- 先回答：策略发生了什么、哪个 component 最值得干预、上一轮改了什么。
- 检查 reward role 是否完备、可达，以及 proxy 是否与外部任务对齐。
- L1：只调整一个 component 的尺度。
- L2：只对一个目标 component 做有方向的数学结构变换，例如 sparse-to-dense、unbounded-to-bounded、state-to-improvement、independent-to-joint。
- L3：在多轮同骨架停滞、L2 无效或长期未刷新 best 时重建主骨架。
- 必要时最多调用一个知识工具：`search_reward_design_knowledge`、`get_skeleton_detail` 或 `get_reward_transformation`。
- 输出八个固定诊断字段，然后立即输出完整 `compute_reward` 代码。

注意：System Prompt 中写有“current 明显差于 best 时，以 best 代码为基础”。但正常 User Prompt 并没有提供 best source code；这是历史实现中的文本—数据不一致。论文复现包保留真实代码，不补造该输入。

### 3.2 System Prompt — English translation

You are a reward-function diagnosis and revision agent. Explain the failure from training evidence first, then select the smallest falsifiable intervention.

Required decision process:

- Treat component statistics as observations, not causal attribution. Interpret them jointly with native score, episode length, termination behavior, and prior interventions.
- First identify what behavior occurred, which component is the best intervention target, and what was changed previously.
- Check whether required reward roles are present and reachable and whether a proxy aligns with the external task.
- Level 1 changes the scale of one component only.
- Level 2 changes the mathematical form of one target component, such as sparse-to-dense, unbounded-to-bounded, state-to-improvement, or independent-to-joint.
- Level 3 rebuilds the main reward structure after persistent same-family stagnation, an ineffective Level-2 transformation, or repeated failure to refresh the best result.
- If necessary, call at most one reward-knowledge tool.
- Emit the eight fixed diagnosis fields and then a complete revised `compute_reward` function.

Historical caveat: the System Prompt says to start from best code when current is much worse, but the normal User Prompt does not contain best source code. The archive preserves this actual implementation mismatch rather than inventing an input that was not used.

### 3.3 User Prompt — 正常迭代的精确模板 / Exact normal-iteration template

````text
# Search objective
- target_score: {task threshold}
- current_score: {native score parsed from the current feedback}
- gap_to_target: {target - current}
- target_achievement_ratio: {current / target}
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: {current score}）
```python
{complete source code of the immediately preceding reward}
```

# 训练反馈（上一轮代码的训练结果）
{complete training_feedback.md}

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
{sections 1, 3, 4, 5 and 7 extracted from environment_card.md}

# 历史记忆
{reward_memory.md, unless memory is disabled by an ablation}
````

English field meanings:

- **Search objective:** target, current native score, remaining gap, and achievement ratio. The target is only a progress criterion.
- **Previous reward code:** the complete source of the immediately preceding reward, not automatically the archived best code.
- **Training feedback:** native outcome, episode behavior, and named component statistics produced by the preceding reward’s policy training/evaluation.
- **Environment facts and expert task profile:** only sections 1, 3, 4, 5, and 7 of the environment card are retained; sections 9–12 are deliberately excluded during reflection.
- **History memory:** the persistent reward-lineage table, unless disabled for an ablation.

### 3.4 明确不存在的字段 / Fields that are not present

正常 paper-v4 Reflection User Prompt **没有**：

- `# 累积迭代记录`
- `# Component delta`
- `# Formula switching guide`
- `# Subagent 调研信号`
- `# Best reward code`

The normal paper-v4 reflection prompt contains none of the five blocks above.

### 3.5 Best Code 的真实行为 / Actual Best-Code behavior

`build_user_prompt(feedback_md, memory_md, previous_code, best_code, ...)` 接收 `best_code` 参数，但历史实现是：

```python
if False:  # 历史最佳代码已移除，用历史记忆表格替代
    pass
```

因此：

- 控制器真实保存 `best/best_reward.py` 与最高 native score；
- 正常 Reflection Agent 看到的是上一轮代码和 memory，不是单独的 best code；
- Best Archive 用于最终返回、best retention、identical-candidate 检查和停止保护；
- 不能在论文中声称“每轮把 Best Code 作为显式 Prompt 输入”。

Therefore, best code is real controller state, but it is not a normal LLM input in paper-v4.

### 3.6 特殊重试 Prompt / Special retry prompts

若生成结果与历史 best 完全相同，控制器在正常模板前加入 duplicate-retry 指令和被拒绝草稿，要求做一个有证据的新修改；这是防止空跑 PPO 的保护分支。

If a generated reward is identical to the archived best, the controller prepends a duplicate-retry instruction and the rejected draft, requiring one evidence-based change. This is a no-op protection branch.

若代码验证失败，控制器在正常模板前加入：

```text
# ⚠️ 上一版代码验证失败
验证错误：{validation error}
请只修复上述代码合规问题，不要重新做诊断，不要改变原定的奖励设计决策。
直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
{failed draft}
```

For validation retry, memory is omitted from that retry message so the model focuses on code compliance rather than a new diagnosis.

---

## 4. Prompt—论文术语对照 / Terminology map

| 论文术语 | Prompt/代码中的真实载体 | English meaning |
|---|---|---|
| Training evidence | `training_feedback.md` | native outcome, episode behavior, and component statistics |
| Persistent Memory | `reward_memory.md` | persistent reward-lineage state shown to the reflection agent |
| Bounded repair | 单目标 + L1/L2/L3 contract + code validation | one-target edit under an explicit intervention hierarchy |
| Native evaluation | unchanged environment reward | external evaluation that the generated shaping reward cannot redefine |
| Best Archive | `best/best_reward.py` and controller metadata | controller-side retention of the highest native-score reward |
| Self-evolution | reward versions and memory updated by verified transitions | evolution of the reward program, not the LLM weights |

## 5. 归档结论 / Archival conclusion

paper-v4 的方法闭环应写成：上一轮 reward 经 fresh PPO 训练后产生 native outcome 与 component evidence；Reflection Agent 结合精简环境事实和 persistent reward memory，选择一次 L1/L2/L3 修复；验证后的 reward 进入下一轮；控制器独立维护 Best Archive。后续 Subagent 等探索不进入本次论文。

The paper-v4 method is therefore a single persistent reward-editing agent driven by policy-training evidence, with controller-side best retention. Later Subagent-based explorations are outside this paper.
