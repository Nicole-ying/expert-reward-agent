# CREATE paper-v4 prompts / CREATE paper-v4 提示词总表

本文档回答两个问题：CREATE 的每一次 LLM 调用分别使用什么 System Prompt 和 User Prompt；这些内容是否能从 paper-v4 实验记录中核对。

This document identifies the System Prompt and User Prompt used by every LLM call in CREATE and states whether each prompt is confirmed by the archived paper-v4 records.

## 1. 审计结论 / Audit conclusion

归档目录 `runs/env_001/paper_v4/seed_*/` 中保存了实际发送给模型的提示词。每条主实验谱系包含以下三类 LLM 调用：

1. Environment Analyzer：读取匿名任务说明和屏蔽官方奖励后的环境接口，生成环境卡片；
2. Initial Reward Generator：读取环境卡片和专家知识，生成 `reward_v1.py`；
3. Reflection and Repair Agent：读取上一版奖励、训练证据和奖励谱系，诊断并生成下一版奖励。

The archived `runs/env_001/paper_v4/seed_*/` directories preserve the prompts actually sent to the model. Every main search lineage contains three LLM stages: environment analysis, initial reward generation, and iterative reflection/repair.

| LLM 阶段 | paper-v4 中实际使用 | System Prompt 来源 | User Prompt 记录示例 |
|---|---:|---|---|
| Environment Analyzer | 是 | `prompts/01_environment_analyzer_prompt.md` | `runs/env_001/paper_v4/seed_0/iter_01/generation/prompt_records/01_environment_analyzer.md` |
| Initial Reward Generator | 是 | `prompts/02_reward_generator_prompt.md` | `runs/env_001/paper_v4/seed_0/iter_01/generation/prompt_records/02_reward_generator.md` |
| Reflection and Repair Agent | 是 | `prompts/reflection_agent_prompt.md` | `runs/env_001/paper_v4/seed_0/iter_02/generation/prompt_records/agent_reflection.md` |
| Evidence Scout / Investigator | 原始 paper-v4 中没有记录 | `pipeline/subagent_investigator.py` 内嵌 | 只在后来启用该模块的实验中产生 `subagent_trace_*.json` |

重要说明：当前 `configs/env001_paper_v4.yaml` 中配置了可选 Investigator，但归档的原始 `paper_v4` 运行没有 `subagent_trace_*.json` 或 `subagent_signal_*.md`。因此，论文现有 paper-v4 数值只能直接证明前三类提示词组成的闭环，不能把后来新增的 Investigator 写成已被这些结果验证的机制。

Important: the current configuration can enable an optional investigator, but the archived paper-v4 runs contain no investigator trace or signal. The reported paper-v4 results therefore directly support only the three-stage prompt chain above.

### 1.1 版本一致性检查 / Version-consistency check

我进一步对归档 `prompt_records` 中的 System Prompt 做了逐字哈希检查。结果是：五个 seeds 使用的版本彼此一致，但实验结束后仓库中的三个主模板都被继续修改过。

I also compared the archived System Prompts by exact hash. All five seeds used a consistent prompt version, but all three source templates were edited after the archived experiment finished.

| 阶段 | 归档版本 | 当前版本 | 主要差异 |
|---|---:|---:|---|
| Environment Analyzer | 181 行 | 238 行 | 当前版新增 derived termination inference、主骨架算子族选择和四步自检框架 |
| Initial Reward Generator | 163 行 | 164 行 | 当前版新增终止前兆、直接进度信号和高维动作效率约束的三项自检 |
| Reflection Agent | 109 行 | 81 行 | 当前版重写为显式信号覆盖审计、量化校准及新的 L3 停滞阈值 |

用于复核的 System Prompt SHA-256 前 16 位：

| 阶段 | 归档 paper-v4 | 当前重跑版本 |
|---|---|---|
| Environment Analyzer | `5fc9a66244be8b01` | `d386aec3f1792d50` |
| Initial Reward Generator | `f735ba54ebb001d8` | `91602129ea024caf` |
| Reflection Agent | `b0409bea95a347e1` | `6c8ad0f99869e358` |

因此，下面每一节同时说明两层内容：`Archived paper-v4` 表示已经产生论文现有结果的真实提示词；`Current rerun` 表示当前代码再次运行时会使用的提示词。二者不能被描述为逐字相同。

Accordingly, each section distinguishes the archived prompt that produced the reported results from the current prompt used by a new rerun. They must not be described as byte-identical configurations.

## 2. Prompt 如何被保存 / How prompts are recorded

`pipeline/common.py::record_prompt` 在每次调用前保存：

```text
# Prompt Record

## System Prompt
<the fixed role and decision policy>

## User Prompt
<the task- and iteration-specific evidence payload>
```

System Prompt 定义角色、证据边界、决策规则和输出格式。User Prompt 注入当前任务、奖励代码、训练结果和历史状态。模型调用最终始终是：

The System Prompt defines the role, evidence boundary, decision policy, and output contract. The User Prompt injects task- and iteration-specific state. Every call is sent as:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
```

## 3. Environment Analyzer / 环境理解模块

### 3.1 System Prompt — 中文原意

角色：强化学习环境理解模块。只负责读懂匿名环境并生成可供人类和后续 LLM 使用的 Markdown 环境卡片，不生成奖励代码。

主要输入：

- 匿名任务描述；
- observation space 和 action space；
- 屏蔽官方奖励实现后的 step source；
- 终止条件和可用 `info` 字段。

必须完成：

1. 描述主任务、次要目标以及不应混淆的目标；
2. 从七类任务族中选择一个 `selected_route_id`；
3. 判断更细的 `dynamics_subtype`；
4. 逐维说明 observation space；
5. 逐项说明 action space；
6. 区分成功、失败、歧义终止和截断；
7. 定义 `compute_reward` 的接口和禁止使用的输入；
8. 区分可用、派生可用和不可用信号；
9. 输出 `expert_task_profile`；
10. 把奖励目标拆成 mandatory、conditional 和 avoid roles；
11. 建立 `role_to_signal_mapping`；
12. 列出初始训练后需要检查的 failure modes。

关键限制：

- 不输出奖励函数或 `reward_v1.py`；
- 不恢复官方奖励，不暴露真实环境名或 Gymnasium ID；
- 不假设 `info["success"]`、`info["failure"]` 或未声明字段存在；
- 不把模板中的奖励组件机械地当作答案；
- 如果成功或失败只能从观测组合推断，必须标记为 `derived_possible`；
- 先确认任务职责，再把职责映射到真实存在的信号和数学算子。

输出固定为 12 节环境卡片：任务目标、任务类型、观测空间、动作空间、终止分析、奖励接口、可用信号、不可用信号、专家任务画像、奖励职责、职责—信号映射以及 failure modes。

归档实验的精确中文原文位于每个 seed 的 `prompt_records/01_environment_analyzer.md`。当前重跑版本位于 `prompts/01_environment_analyzer_prompt.md`。当前版在归档版基础上新增了间接终止推断、按任务族选择主信号算子，以及静止、完成后刷分和信号稀疏性的自检。

### 3.2 System Prompt — English translation

Role: You are an RL environment-understanding module. Your only job is to understand an anonymized environment and produce a Markdown environment card that is readable by humans and reusable by downstream LLMs. Do not generate reward code.

Required inputs include the anonymized task description, observation and action spaces, a step source with the official reward masked, termination conditions, and declared `info` fields.

You must:

1. state the primary objective, secondary objectives, and objectives that must not be confused;
2. select exactly one coarse `selected_route_id` from the seven task families;
3. infer a more specific `dynamics_subtype`;
4. describe every observation index;
5. describe every action or action dimension;
6. separate success-like, failure-like, ambiguous termination, and truncation;
7. define the `compute_reward` interface and forbidden inputs;
8. separate usable, derived-possible, and unavailable signals;
9. produce an `expert_task_profile`;
10. decompose reward responsibilities into mandatory, conditional, and avoid roles;
11. map every role to observable signals;
12. list failure modes to inspect after initial training.

Do not generate `reward_v1.py`, reconstruct the official reward, reveal the real benchmark identity, invent undeclared observations or `info` fields, or mechanically select reward components from a template. When success or failure can only be inferred from a combination of observations, label the path `derived_possible`.

Return a twelve-section environment card covering the task, spaces, termination semantics, reward API, usable signals, task profile, role decomposition, role-to-signal mapping, and post-training failure checks.

### 3.3 User Prompt — 实际模板

```text
ANONYMIZED_TASK_SPEC:
{contents of envs/env_001/task_spec_anonymized.yaml}

MASKED_STEP_SOURCE:
{contents of envs/env_001/masked_step_source.py}
```

中文解释：User Prompt 不要求模型设计奖励，只提供任务事实和被屏蔽的接口代码。官方奖励的位置显示为 `<OFFICIAL_REWARD_MASKED>`，防止模型复制标准答案。

English: the User Prompt supplies only task facts and the masked interface. The official reward location is represented by `<OFFICIAL_REWARD_MASKED>` so that the model cannot copy the benchmark reward.

## 4. Initial Reward Generator / 初始奖励生成模块

### 4.1 System Prompt — 中文原意

角色：奖励函数生成模块。根据环境卡片与专家知识生成第一版可执行奖励 `reward_v1.py`。

主要决策顺序：

1. 从环境卡片读取 `expert_task_profile`；
2. 确认 mandatory、conditional 和 avoid reward roles；
3. 检查每个 role 是否有真实可用的 observation、action 或 `info` 信号；
4. 为选中的 role 选择合适的公式算子；
5. 最后才编写奖励函数。

设计规则：

- 环境卡片中的信号事实高于专家模板；
- 专家知识是设计启发，不是固定答案；
- v1 优先包含一个稠密的主要学习信号和必要的轻量健康约束；
- 推荐 2–4 个职责明确的组件，但不机械限制组件数；
- 二值事件过稀疏时优先使用连续、bounded 或 soft proxy；
- 不让惩罚项压制探索，不让重复信号同时获得大权重；
- v1 默认不引入复杂课程、强门控或不必要的动作代价；
- 检查速度刷分、静止生存、接触刷分和原地活动等 reward hacking 风险。

代码硬约束：

- 唯一函数签名为 `compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)`；
- 不允许 import、class、额外函数、`try/except`、`eval/exec/open`；
- 不使用 `original_reward` 或未声明字段；
- 返回 `(float(total_reward), components)`；
- `components` 只包含真正进入总奖励的命名分量；
- 第一个 Python code block 必须是完整可执行函数。

输出包括奖励代码和简短设计说明：任务族、所选职责、职责—信号映射、公式算子、排除的职责、未使用终止奖励的理由以及下一轮应观察的 failure modes。

归档实验的精确中文原文保存在每条谱系第一轮的 `prompt_records/02_reward_generator.md`。当前重跑版本位于 `prompts/02_reward_generator_prompt.md`。两版主体相同；当前版额外要求检查终止条件是否有前兆软信号、任务是否存在直接进度信号，以及高维动作任务是否缺少轻量效率约束。

### 4.2 System Prompt — English translation

Role: You are the initial reward-generation module. Use the environment card and expert reward context to produce the first executable reward, `reward_v1.py`.

Follow this order: read the task profile; identify mandatory, conditional, and excluded reward roles; verify that every selected role has a declared signal; select a formula operator for each role; only then write code.

Environment facts override generic templates. Expert context provides design priors rather than a fixed solution. The initial reward should prioritize one dense task-driving signal plus only the health or safety constraints that are necessary. Prefer continuous, bounded signals to rarely triggered binary conditions; keep component scales comparable; prevent penalties from suppressing exploration; avoid duplicate signals and obvious reward-hacking shortcuts; and postpone complex curricula, strong gating, and efficiency costs unless the task requires them.

The output must contain exactly one complete `compute_reward` implementation in the first Python block. Do not import packages, define classes or helper functions, use exception handling, call `eval`, `exec`, or `open`, use the original environment reward, or invent undeclared inputs. Return a scalar total and a dictionary containing only the components that appear in that total.

After the code, explain the selected task family, reward roles, signal mapping, formula operators, excluded roles, terminal-signal decisions, deferred responsibilities, and failure modes to inspect after training.

### 4.3 User Prompt — 实际模板

```text
# environment_card.md
{Environment Analyzer 的输出}

# expert_reward_context.md
{由 RAG/knowledge_base 构造的任务相关奖励设计知识}

[# masked_step_source.py
{仅当 include_masked_step_in_reward_generator=true 时加入；paper-v4 默认不加入}]

[{restart_context，只有重新开始一条失败搜索方向时加入}]
```

English template:

```text
# environment_card.md
{output of the Environment Analyzer}

# expert_reward_context.md
{task-relevant reward-design context built from the knowledge base}

[# masked_step_source.py
{included only when the debug flag is enabled; disabled in paper-v4}]

[{restart context, only when restarting a failed search direction}]
```

如果生成代码验证失败，会在同一个 User Prompt 前后追加以下 repair instruction：

If code validation fails, the following repair instruction is appended:

```text
# Validation repair
具体错误 / validation error: {validation_retry}
只修复代码合规问题，不重新分析环境，不改变原定奖励设计。
Fix only code-compliance problems. Do not re-analyze the environment or change the intended reward design.

# Invalid previous draft
{failed draft}
```

## 5. Reflection and Repair Agent / 诊断与修复 Agent

### 5.1 System Prompt — 中文原文结构

角色：奖励函数诊断与修订 Agent。必须先用训练证据解释失败，再选择最小且可验证的干预。正常模式每轮只修改一个主要目标，只有明确进入 REBUILD MODE 时才允许更换主信号骨架。

证据边界：

- 只根据环境事实判断，不猜测环境身份或变量；
- 正确解释 `episode_sum_mean`、`signed_share`、`magnitude_share` 和 `active_rate`；
- 组件统计是观察证据，不是因果贡献；
- 必须联合原生 score、episode length、terminated/truncated 分布和历史修改；
- 不直接比较具有不同时间语义的奖励分量；
- 不因任务语义关键词而凭空增加奖励职责。

固定决策顺序：

1. **信号覆盖审计**：检查终止模式、未使用观测、信号缺口和僵尸组件；
2. **行为与历史诊断**：判断快速失败、超时徘徊或 reward exploit；选一个最值得干预的组件；检查上一轮修改是否有效；
3. **选择干预层级**：
   - L1：职责和公式合理，仅修复系数或阈值；
   - L2：针对稀疏、无界、持续刷分、全局约束干扰、乘积塌缩、proxy 错位或信号缺失进行结构变换；
   - L3：同一骨架连续停滞或 L2 无实质改善时重建主骨架；
4. **设计校准**：限制新惩罚相对主信号的逐步尺度、hinge 阈值和 gate 下界；
5. 输出八个固定字段，再输出完整奖励代码。

八个字段为：`evidence`、`behavior_diagnosis`、`signal_completeness`、`selected_level`、`selected_intervention`、`falsifiable_hypothesis`、`expected_next_round` 和 `main_risk`。

归档 paper-v4 的完整中文 System Prompt 位于每轮 `prompt_records/agent_reflection.md`，共 109 行。当前重跑版本位于 `prompts/reflection_agent_prompt.md`，共 81 行。上面的“信号覆盖审计”和量化设计校准主要描述当前版；归档版的真实决策策略单独总结在第 5.5 节。

### 5.2 System Prompt — English translation

Role: You are a reward-diagnosis and revision agent. Explain the failure from training evidence before selecting the smallest testable intervention. In normal mode, modify only one primary target per iteration. Replace the main reward scaffold only when the User Prompt explicitly marks REBUILD MODE.

Evidence boundary:

- reason only from declared environment facts;
- interpret component sum, signed share, magnitude share, and activation rate correctly;
- treat component statistics as observations rather than causal contributions;
- combine them with native score, episode length, termination/truncation, and edit history;
- do not directly compare components with different temporal semantics;
- do not invent a missing reward role merely from a semantic keyword in the task description.

Decision order:

1. audit termination modes, unused observations, missing signals, and dead components;
2. diagnose rapid failure, timeout wandering, or reward exploitation; select one primary component; inspect whether the previous intervention worked;
3. choose L1 scale repair, L2 mathematical restructuring, or L3 scaffold rebuilding using the declared triggers;
4. calibrate the new penalty, hinge threshold, and gate against the observed per-step task signal;
5. output the eight fixed diagnostic fields followed immediately by complete reward code.

The eight fields are `evidence`, `behavior_diagnosis`, `signal_completeness`, `selected_level`, `selected_intervention`, `falsifiable_hypothesis`, `expected_next_round`, and `main_risk`. The hypothesis and quantitative expectation must be falsifiable by the next training run.

### 5.3 User Prompt — 正常迭代模板

```text
# 1. Search objective
- target_score: {task threshold}
- current_score: {native score of R_t}
- gap_to_target: {target - current}
- target_achievement_ratio: {current / target}

# 2. 上一轮奖励函数代码
{complete source code of R_t}

# 3. 累积迭代记录
{previous iteration, score, best, diagnosis, edit and outcome records}

# 4. Component delta
{changes in named component statistics relative to the preceding evaluation}

# 5. 本轮训练反馈
{native score, episode lengths, termination distribution,
 component episode_sum_mean, signed_share, magnitude_share and active_rate}

[# 5.5. Subagent 调研信号
{optional investigator evidence; absent from archived paper-v4}]

# 6. 环境事实
{compact environment card}

# 7. Formula switching guide
{task-relevant formula transformations and anti-patterns}

# 8. 历史记忆
{reward lineage memory}
```

English explanation: the User Prompt is the complete state observed at reward-evaluation iteration `t`. It contains the current native objective gap, active reward code, prior transitions, changes in reward-component behavior, the latest training evidence, task facts, a compact operator guide, and lineage memory. The model must return the next executable reward `R_{t+1}`.

### 5.4 特殊 User Prompt / Special User Prompt prefixes

**REBUILD MODE / 重建模式**

```text
# REBUILD MODE
系统已经接受 Level 3 重建建议。不要受上一轮代码结构约束；
根据完整历史选择新的主信号框架，并避开已经失败的路径。

The system accepted a Level-3 rebuild. Design a new primary reward scaffold
from the full history instead of remaining constrained by the previous code.
```

**Duplicate retry / 重复奖励重试**

```text
# Duplicate reward retry
The previous draft is semantically identical to the trained reward.
Re-analyze the evidence and implement one materially different intervention.
Do not merely rename variables or comments.

# Rejected duplicate draft
{duplicate code}
```

**Validation retry / 代码验证重试**

```text
# 上一版代码验证失败
错误信息 / error: {validation_retry}
这是代码格式修复：不要重新诊断、不要调用工具、不要改变原定修改方向。

This is a code-format repair. Do not re-diagnose, call tools, or change the
selected intervention.

# 被截断或无效的上一版草稿
{failed draft}
```

### 5.5 Archived paper-v4 Reflection policy / 归档实验真实修复策略

归档 paper-v4 的 Reflection System Prompt 使用以下顺序：

1. **行为与历史诊断**：回答 Agent 发生了什么、哪个奖励组件最值得干预、上一轮修改了什么；
2. **信号完备性检查**：检查任务进展、稳定/安全约束和任务完成信号是否完备且可达；
3. **L1 尺度修复**：一次只调整一个组件的系数；`|penalty/progress| > 0.5` 只作为经验检查触发器，而不是通用因果阈值；
4. **L2 结构变换**：包含 sparse-to-dense、unbounded-to-bounded、state-to-improvement、global-to-local gate、independent-to-joint、product-to-noncollapsing、persistent-to-transition、proxy-to-completion alignment、coupled-to-diagnostic components 和 dense-to-task-event；
5. **L3 骨架重建**：归档版使用 `历史最佳 < target×0.25` 等停滞条件，而当前版使用了重新整理后的条件；
6. **工具调用**：仅在根因不确定或多个 L2 方案难以区分时调用一个最相关工具；
7. **输出**：同样要求八个诊断字段、可证伪预测和完整的下一版奖励代码。

The archived paper-v4 Reflection Prompt followed this order: behavioral and historical diagnosis; signal-completeness checking; single-component L1 scale repair; one targeted L2 mathematical transformation; L3 rebuilding under the archived stagnation criteria; at most one relevant knowledge tool when needed; and eight diagnostic fields followed by complete reward code.

The archived L2 operator set covered sparse-to-dense, unbounded-to-bounded, state-to-improvement, global-to-local gating, independent-to-joint objectives, non-collapsing replacement of products, transition-based event rewards, proxy-to-completion alignment, diagnostic decomposition, and conversion of dense proxy plateaus into task-event signals.

归档版与当前版的核心故事相同：训练证据定位故障，每轮选择一个主要干预，并要求下一轮训练验证预测。差异主要在诊断清单、经验阈值和 L3 触发条件。因此，重新跑实验时必须记录实际 prompt 版本，不能只写“paper-v4 prompt”。

The archived and current prompts share the same scientific idea—localize a failure from training evidence, choose one primary intervention, and verify its prediction in the next run—but differ in diagnostic checklists, heuristic thresholds, and L3 triggers. Every rerun must therefore record the exact prompt version rather than referring only to a generic “paper-v4 prompt.”

## 6. Evidence Scout / Investigator（当前可选模块）

该模块在当前代码中存在，但没有出现在原始 `paper_v4` 记录中。它只能报告训练证据，不能替主 Agent 选择奖励修改。

This module exists in the current code but is absent from the archived paper-v4 records. It may summarize evidence but is forbidden from choosing an edit for the main agent.

### 6.1 System Prompt — original English

```text
You are an EVIDENCE SCOUT. Read the training data below and produce a compact
JSON signal (~400-600 chars total across all fields).

YOU DO NOT make decisions or suggest reward edits. You report what you
OBSERVED in the data. The reward designer (a separate LLM) owns all decisions.

Output valid JSON with these fields:
- key_findings
- component_anomalies
- training_dynamics
- signal_quality
- confidence: low/medium/high

Every claim must reference a metric in the data. Do not propose coefficients,
operators, or edit strategies. If data is sparse, say so and use low confidence.
```

### 6.2 System Prompt — 中文翻译

```text
你是“证据侦察员”。阅读下面的训练数据，输出一个紧凑的 JSON 信号。

你不能作出奖励修改决策，也不能建议如何编辑奖励。你只报告数据中实际观察到的事实；
奖励设计决策属于另一个主 Agent。

输出字段：关键发现、异常奖励组件、训练动态、信号质量和置信度。
每个判断必须引用可见指标；不得建议系数、算子或编辑策略。
数据不足时必须明确说明，并把置信度设为 low。
```

### 6.3 User Prompt — bilingual template

```text
Training data for the most recently completed reward iteration:
最近一次奖励评估的训练数据：

=== Training Summary / 训练摘要 ===
{training_summary.json}

=== Component Dynamics / 奖励分量时间动态 ===
{checkpoint component statistics}

=== Training Feedback / 最终策略反馈 ===
{training_feedback.md, truncated to 3000 chars}

=== Previous Reward Code / 上一版奖励代码 ===
{previous reward, truncated to 3000 chars}

Output facts only, with no recommendations.
只输出事实，不提供修改建议。
```

## 7. 消融条件怎样改变 Prompt / How ablations modify prompts

### Score-only feedback

System Prompt 不变，但 User Prompt 中的组件诊断表被删除，只保留原生任务结果和评估分布。历史记忆也只保留迭代编号、奖励骨架、score、best、delta 和 episode length。

The System Prompt is unchanged. The User Prompt removes the component evidence table and retains only the native outcome and evaluation distribution. Memory is reduced to iteration, scaffold, score, best, delta, and episode length.

### Coarse or EUREKA-style feedback

System Prompt 不变。User Prompt 保留原生分数和每个奖励组件的 `episode_sum_mean`，但删除 `signed_share`、`magnitude_share` 和 `active_rate`。

The System Prompt is unchanged. The User Prompt retains the native score and each component's mean episodic value but removes signed share, magnitude share, and activation rate.

### Unconstrained editing

User Prompt 的证据内容保持不变，System Prompt 改为 `prompts/reflection_agent_unconstrained_prompt.md`。它允许根据证据同时调整系数、改变数学形式、增加或删除组件或重构整体结构，不再要求单目标 L1/L2/L3 合同。

The evidence payload remains unchanged, while the System Prompt switches to `prompts/reflection_agent_unconstrained_prompt.md`. The editor may tune coefficients, change mathematical forms, add or remove components, or rebuild the reward without the single-target L1/L2/L3 contract.

### Independent generation baseline

当前公平 baseline 脚本 `scripts/run_independent_baseline.sh` 对每个候选重复使用与 CREATE 相同的 Environment Analyzer 和 Initial Reward Generator 路径。训练结果不进入下一个候选的 User Prompt，因此不存在 Reflection Prompt、Persistent Memory 或跨候选修复。

The prompt-matched baseline reuses CREATE's Environment Analyzer and Initial Reward Generator for each candidate. A candidate's training result is never inserted into the next candidate's User Prompt, so there is no reflection prompt, persistent memory, or cross-candidate repair.

注意：仓库中存在 `prompts/02_reward_generator_prompt_independent_baseline.md`，但当前 prompt-matched baseline 脚本没有引用它。它是保留的旧模板，不应被误认为当前公平 baseline 的实际 System Prompt。

Note: `prompts/02_reward_generator_prompt_independent_baseline.md` is a retained legacy template and is not referenced by the current prompt-matched baseline script.

## 8. Tool definitions 不是 System/User Prompt

Reflection Agent 可以通过 API 的 `tools` 字段访问三个检索工具：

- `search_reward_design_knowledge(query)`：查询相似失败模式和经验修复；
- `get_skeleton_detail(skeleton_name)`：查询奖励骨架的数学性质和风险；
- `get_reward_transformation(query)`：查询结构变换的原理、风险和验证目标。

These schemas are sent through the model API's `tools` field rather than concatenated into the System or User Prompt. Tool results are appended as tool-role messages before the model produces its final reward code.

## 9. 未进入 paper-v4 主实验的旧 Prompt / Legacy prompts not used by paper-v4

以下模板属于早期多阶段 pipeline，没有出现在归档 `paper_v4` 的实际 prompt records 中：

- `prompts/02_reward_architect_prompt.md`；
- `prompts/03_reward_coder_prompt.md`；
- `prompts/03_reward_revision_prompt.md`；
- `prompts/04_analysis_prompt.md`；
- `prompts/01_environment_analyzer_prompt_ablation_facts_only.md`；
- `prompts/02_reward_generator_prompt_independent_baseline.md`。

These templates remain for historical or alternative pipelines but are not part of the archived paper-v4 prompt chain. They should not be described as CREATE's reported main-method prompts unless a new experiment explicitly invokes them.

## 10. 一句话理解整个提示词闭环 / One-sentence prompt flow

中文：环境分析 Prompt 把匿名任务变成受约束的环境卡片；初始生成 Prompt 把环境卡片变成可执行的 `R_0`；修复 Agent Prompt 把训练证据、当前奖励和历史谱系变成下一次可验证的 `R_{t+1}`。

English: the environment prompt converts an anonymized task into a constrained environment card; the initial-generation prompt converts that card into executable `R_0`; and the repair-agent prompt converts training evidence, the active reward, and its lineage into the next testable `R_{t+1}`.
