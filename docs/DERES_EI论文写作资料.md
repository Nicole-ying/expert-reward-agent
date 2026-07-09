# DERES_EI论文写作资料

> 由 Word 初稿《DERES_EI论文中文初稿_故事版.docx》转存为 Markdown，便于在仓库中持续修改、版本管理和后续论文写作。
> 文中“先导实验”表述用于论文故事线与实验设计讨论，正式投稿前应替换为冻结版本的最终 5 seed × 10 iteration 结果。

DERES：基于结构化诊断的大语言模型奖励函数自进化搜索方法

——面向 EI 会议投稿的中文论文初稿（故事版 / 可继续替换实验数据）

方法英文名建议：Diagnosis-guided Expert Reward Evolution Search (DERES)

> 版本说明：本稿依据当前仓库实验记录、近期相关研究与已修改的 Expert Schema 框架整理。文中“先导实验”用于论证故事线，正式投稿前需要用冻结后的 5 seed × 10 iteration 实验替换为最终表格和图。


## 摘要

奖励函数设计是强化学习应用中的关键瓶颈。近年来，大语言模型（Large Language Models, LLMs）被用于自动生成奖励函数代码，显示出利用语言先验和代码生成能力降低人工奖励设计成本的潜力。然而，现有方法往往将奖励设计视为一次性生成或多候选采样问题：失败的奖励函数通常被直接淘汰，而没有被转化为后续搜索的有效证据。这使得 LLM 奖励设计在复杂控制任务中容易出现高方差、重复失败、奖励代理目标错位和搜索不可解释等问题。

为此，本文提出 DERES（Diagnosis-guided Expert Reward Evolution Search），一种基于结构化诊断的大语言模型奖励函数自进化搜索框架。DERES 首先将匿名环境描述抽象为环境卡片、任务画像、奖励职责与职责-信号映射；随后由奖励生成模块产生初始奖励函数，并通过 PPO 训练获得策略表现。与仅使用总分反馈不同，DERES 将外部评价分数、episode 长度、终止统计、奖励组件均值、绝对贡献和激活率等训练证据组织为结构化诊断输入，由 Diagnostic Reflection Agent 判断失败行为、选择干预层级，并对单个目标奖励组件执行可追踪的局部修复。Reward Memory 与 Best Reward Archive 分别保存历史搜索轨迹与最优奖励，Failure Recovery 机制处理无效代码、重复候选和长期停滞，从而形成“失败奖励—结构化诊断—局部修复—再训练”的闭环自进化过程。

在着陆控制类 Env_001 的先导实验中，DERES 展现出将失败初始奖励逐步修复为可解奖励的趋势；在连续步态控制 Env_002 中，现有记录显示多个随机种子均能在少量迭代内达到解决阈值。本文进一步设计 LLM-Once、Score-Only Evolution、Independent-K、w/o Failure Recovery 和 w/o Expert Schema 等基线与消融实验，以验证结构化诊断、失败恢复和专家 Schema 对搜索成功率与搜索效率的作用。本文的核心观点是：LLM 辅助奖励设计不应只被视为“生成奖励代码”，而应被建模为一个可诊断、可修复、可追踪的奖励函数自进化过程。

> 关键词：强化学习；奖励函数设计；大语言模型；自进化搜索；结构化诊断；PPO；智能体

## 1 引言

强化学习通过最大化累积奖励来学习策略，但其性能高度依赖奖励函数的设计质量。在许多实际控制任务中，人工奖励设计不仅耗时，而且容易出现代理目标偏差：奖励函数可能鼓励智能体靠近目标但不完成任务，可能过分惩罚动作导致策略保守，也可能奖励某些局部状态使策略产生“刷分”行为。随着大语言模型在代码生成和语义理解方面的能力提升，利用 LLM 自动生成奖励函数成为近年来强化学习自动化设计中的重要方向。

已有研究如 Eureka 利用 LLM 的代码生成与 in-context improvement 能力进行奖励代码优化，并在多个强化学习任务中获得较好效果；REvolve 则引入人类反馈引导奖励演化；CARD 使用动态反馈与轨迹偏好评估以减少人工干预；ERFSL 将 LLM 作为奖励函数搜索器，对多目标任务中的奖励组件和权重进行迭代调整；LLM2Reward 等工作进一步关注奖励观测空间的演化与历史缓存。这些研究共同说明：LLM 确实可以参与奖励函数自动设计，但也暴露了一个尚未充分解决的问题——失败奖励函数如何被有效利用。

在多数 LLM 奖励设计框架中，一个奖励候选若训练失败，往往只作为低分样本被淘汰，或仅通过总分反馈促使下一轮重新生成。这种方式忽略了失败训练过程中的大量结构性信息。例如，episode 是否过短可以提示策略快速失败；奖励组件是否长期不激活可以提示稀疏 proxy 无效；某个惩罚组件绝对贡献过大可以提示探索被压制；总分提高但任务未完成可能提示代理目标错位。换言之，失败奖励并非无用样本，而是包含了下一轮奖励修复所需的诊断证据。

本文基于这一观察提出 DERES。不同于将 LLM 作为一次性奖励代码生成器，DERES 将 LLM Agent 定义为奖励函数搜索过程中的诊断与修复执行者。该 Agent 的输入不是单纯的任务描述，而是环境卡片、结构化训练反馈、奖励组件统计、历史记忆和最优奖励档案；其动作不是直接控制环境，而是对奖励函数代码执行可解释的局部干预。通过这种设计，奖励函数设计被重新建模为一个“失败—诊断—修复—验证”的闭环过程。

本文面向 EI 会议的贡献可概括为三点。第一，提出一种诊断引导的奖励函数自进化闭环，将失败奖励训练结果转化为后续奖励修复的可操作证据。第二，设计组件级结构化反馈机制，将外部任务表现、episode 统计和奖励组件统计映射为奖励函数层面的诊断问题。第三，提出带有 Reward Memory、Best Reward Archive 和 Failure Recovery 的顺序搜索框架，使 LLM Agent 的奖励修改过程具备可追踪性、可归因性和稳定性。

## 2 相关工作与差异定位

### 2.1 LLM 辅助奖励函数设计

Eureka 是 LLM 奖励设计方向的重要代表，它将奖励函数写作视为代码生成与进化优化问题，利用 LLM 对奖励代码进行多轮改进。其重点在于证明 LLM 可以生成高质量奖励函数，并通过大规模环境验证其泛化能力。与 Eureka 相比，本文不强调大规模多环境覆盖，而强调奖励搜索中的失败诊断机制：当一个奖励候选训练失败时，DERES 不只是重新生成，而是分析该失败对应的组件、尺度、激活率和行为表现，并据此选择局部干预。

REvolve 将 LLM 与人类反馈结合，用人类隐性知识引导奖励演化。该思路适用于难以精确定义“好行为”的任务，但需要人类反馈参与。DERES 的差异在于反馈来源：本文不依赖人工自然语言评价，而从 PPO 训练结果和奖励组件统计中构造结构化诊断证据。换言之，DERES 试图把策略训练本身变成奖励函数修改的反馈来源。

CARD、ERFSL、LLM2Reward 等近期工作进一步从不同角度探索了 LLM 奖励搜索。CARD 关注动态反馈与轨迹偏好，ERFSL 强调对奖励组件和权重的迭代搜索，LLM2Reward 关注奖励观测空间的历史缓存与演化。与这些方法相比，DERES 的核心不是提出一个更大的搜索空间，而是将搜索空间中的失败轨迹转化为可诊断、可修复、可追踪的奖励演化过程。

### 2.2 奖励函数失败与奖励黑客

奖励函数并不总是准确表达任务目标。若奖励设计不当，智能体可能学会利用代理指标，而非完成真实任务。例如，持续状态奖励可能导致占据某个局部好状态，速度奖励可能导致快速冲刺后失败，接触奖励可能导致反复触碰而不完成任务。已有研究通常从 reward hacking 或 reward misalignment 的角度分析这些问题，而 DERES 的关注点是：当这些问题在训练日志中出现时，LLM Agent 如何识别其证据，并采取最小、可验证的奖励修复动作。

因此，DERES 并不把 reward hacking 仅作为失败现象，而将其转化为 reward repair 的触发条件。一个组件的 active rate、episode_sum_mean、abs_mean、magnitude_share 等统计量，虽然不能直接说明因果贡献，但可以作为诊断线索，帮助 Agent 判断是尺度问题、信号缺失、数学形态错位，还是整体骨架需要重建。

### 2.3 本文方法的边界

本文不声称提出 fully autonomous agent。DERES 中的 Agent 是一个奖励函数搜索 Agent，其动作空间是奖励代码修改，感知输入是环境卡片、训练反馈和历史记忆，目标是提高外部评价得分。这样的定义更加符合当前系统能力，也避免对自主规划能力过度夸大。本文的创新边界在于：如何让 LLM Agent 在奖励函数搜索任务中利用失败训练证据进行诊断式自进化。

## 3 方法：DERES 奖励函数自进化框架

### 3.1 总体框架

DERES 的基本流程包括五个阶段：环境抽象、初始奖励生成、策略训练、结构化诊断和奖励修复。给定匿名环境描述，Environment Analyzer 首先抽取任务目标、观测空间、动作空间、终止条件和 reward 接口约束，生成 Environment Card。随后 Expert Schema Prompting 提供任务画像、奖励职责与 formula operator，使 Reward Generator 在不使用官方奖励的前提下生成第一版奖励函数。PPO Trainer 使用该奖励函数训练策略，Evaluator 以被屏蔽的环境原始奖励作为外部评价指标。若当前奖励未达到目标阈值，则 Diagnostic Reflection Agent 根据结构化反馈和历史记忆选择局部干预，修改奖励代码并进入下一轮训练。

该流程的关键不在于 LLM 是否一次写出正确奖励，而在于每一次失败都被保留下来，并转化为下一轮搜索证据。若策略快速失败，Agent 可能降低过强惩罚或增强过程引导；若策略长期存活但低分，Agent 可能增加任务进展信号；若某一奖励组件 active rate 过低，Agent 可能将稀疏条件替换为连续 proxy；若当前轮差于历史最佳，best archive 确保系统不会丢失已找到的有效奖励。

### 3.2 环境卡片与 Expert Schema

在最新框架中，本文不再将专家知识表述为 RAG 检索贡献，而采用 Expert Schema Prompting。环境卡片不仅记录 observation/action/termination，还输出 task profile、reward role decomposition 和 role-to-signal mapping。也就是说，环境理解阶段不直接推荐 reward skeleton，而是先判断任务需要哪些奖励职责，例如 sustained progress、healthy posture、safe contact、light energy regularization 等，再说明这些职责可以由哪些 obs/action 信号支持。

Expert Schema 只提供少量公式算子和动力学类型模板，包括 dense_state_signal、bounded_signal、improvement_delta、potential_based_shaping、quadratic_penalty、soft_health_gate、joint_condition_proxy 和 curriculum_weighting 等。Reward Generator 必须遵循“先选 role，再选 signal，再选 formula operator，最后生成代码”的顺序。这一设计避免了按 benchmark 名称硬套奖励模板，也减少了不适配任务类型的错误先验。

### 3.3 结构化训练反馈

每轮 PPO 训练后，DERES 不仅记录外部平均回报，还统计 episode length、终止/截断情况和各奖励组件的表现。每个组件可以包含 mean、abs_mean、episode_sum_mean、active_rate、magnitude_share 等指标。与只给 LLM 一个总分相比，这些组件级证据可以帮助 Agent 判断奖励函数内部哪个部分更可能导致失败。

需要强调的是，组件统计不是严格因果解释。一个组件占比高，并不意味着它一定是失败原因；一个组件均值低，也不意味着它无用。因此 DERES 的 Reflection Prompt 明确要求 Agent 结合外部 score、episode length、历史修改和组件数学形态共同判断。这样既利用了统计证据，又避免把数值占比机械解释为因果贡献。

### 3.4 Diagnostic Reflection Agent

Diagnostic Reflection Agent 是 DERES 的核心。它每轮需要回答三个问题：第一，这个 agent 发生了什么，即当前策略表现对应何种失败行为；第二，哪个奖励组件或缺失职责最值得干预；第三，上一轮修改了什么以及实际效果如何。基于这些信息，Agent 选择一个干预层级。

Level 1 是尺度修复，适用于组件职责正确但权重过强或过弱的情况；Level 2 是数学结构变换，适用于信号稀疏、无界支配、状态占据刷分、目标补偿等问题；Level 3 是主结构重建，适用于同一结构多轮停滞或局部修复无效的情况。除 Level 3 外，Agent 每轮只修改一个目标组件，从而保持修改动作的可归因性。

### 3.5 Memory、Best Archive 与 Failure Recovery

Reward Memory 记录每轮奖励结构、得分、是否刷新 best、修改动作和关键诊断。它的作用不是替代外部评价，而是防止 Agent 重复无效修改。例如，如果上一轮已经降低某个惩罚系数但得分无改善，下一轮不应继续重复同一动作。Best Reward Archive 则保存历史最佳奖励和策略，避免后续探索把已经找到的有效奖励覆盖掉。

Failure Recovery 包括三类机制：无效代码修复、重复奖励候选检测和长期停滞后的 fresh restart。该机制是 DERES 能够稳定搜索的重要组成部分。对于 EI 会议论文而言，这部分具有较强工程创新价值，因为它把 LLM 输出的不稳定性转化为可控的搜索流程，而不是简单依赖 LLM 一次性正确生成代码。

## 4 实验设计与研究问题

### 4.1 研究问题

本文拟围绕以下问题展开实验。

RQ1：DERES 是否优于一次性 LLM 奖励生成？对应比较 LLM-Once 与 DERES，观察 best return、Success@K 和 initial-to-best improvement。

RQ2：结构化诊断是否提供有效搜索方向？对应比较 Score-Only Evolution 与 DERES。Score-Only 保留顺序迭代，但只提供总分、长度和终止结果，不提供组件统计。

RQ3：失败恢复机制是否是搜索成功的关键？对应比较 DERES 与 w/o Failure Recovery，观察 invalid reward、duplicate reward、fresh restart 和最终解决率。

RQ4：顺序诊断式搜索是否优于等预算独立采样？对应比较 DERES 与 Independent-K，二者使用相同数量的 LLM 候选和 PPO 训练预算。

RQ5：框架是否能迁移到连续控制任务？对应在 Env_002 上验证 DERES 是否能在少量迭代内达到 solved threshold。

### 4.2 主实验环境

Env_001 是一个着陆/目标接近类控制任务，具有位置、速度、角度、接触等状态信号。该任务适合验证 DERES 的失败恢复能力，因为初始奖励容易出现靠近目标但不稳定、速度惩罚过强、稀疏接触条件不触发等问题。

Env_002 是连续步态控制任务，具有更高维状态和连续动作，适合验证框架在连续控制任务上的适应性。与 Env_001 不同，Env_002 的核心奖励职责更加接近持续前进和姿态稳定，因此可以检验 Expert Schema 和结构化诊断是否能迁移到不同动力学类型。

### 4.3 对比方法

| 方法 | 作用 |
| --- | --- |
| Official-Reward PPO | 使用官方奖励训练，仅作为可解性与上界参考，不作为公平奖励设计对手 |
| LLM-Once | 只生成一次奖励函数，不允许迭代 |
| Score-Only Evolution | 保留迭代，但只提供总分反馈，不提供组件统计 |
| Independent-K | 独立生成 K 个奖励候选，训练后选最优，不使用顺序诊断 |
| w/o Failure Recovery | 去掉无效代码修复、重复候选恢复和 fresh restart |
| w/o Expert Schema | 不使用任务模板和公式算子，仅依赖环境事实 |
| DERES | 完整框架 |

### 4.4 评价指标

主要指标包括 Best@K、Success@K、FirstSolveIter、Initial-to-Best Improvement、Reward Rescue Rate、Invalid Code Count、Duplicate Reward Count 和总 PPO timesteps。Official Reward PPO 仅用于说明环境可解性和训练上界，不能作为 DERES 必须超过的目标。

## 5 先导实验发现与预期结果表达

需要说明的是，本节使用“先导实验发现”作为论文故事线的依据，不应在正式投稿时直接视为最终结果。正式论文需要冻结 prompt、代码、模型版本、PPO 超参数和评估 seed 后，重跑 5 seed × 10 iteration 主实验。

### 5.1 Env_001 的失败恢复现象

现有 Env_001 记录显示，DERES 并非在所有 seed 中一次生成即成功，而是在多个 seed 中通过第 3、8、10 轮找到最佳奖励。这一现象非常重要，因为它说明 DERES 的价值不只是初始奖励生成，而是奖励函数在训练反馈驱动下的持续修复。换言之，Env_001 应作为本文“失败奖励可被救活”的主实验环境。

论文中可以重点展示三类图：第一，多 seed 的 reward search trajectory，显示每轮候选得分和 best-so-far；第二，initial-to-best paired improvement，显示每个 seed 从 Iter1 到 best 的提升；第三，failure recovery case study，展示一个 seed 中从失败 reward 到 solved reward 的具体修改路径。

### 5.2 Env_002 的连续控制验证

Env_002 现有记录显示多个 seed 在少量迭代内达到 solved threshold。这一结果适合作为辅助实验，说明 DERES 不只适用于着陆类任务，也能扩展到连续步态控制。但 Env_002 不应作为唯一主故事，因为其初始奖励可能已经较强，容易被审稿人质疑是 forward velocity 先验带来的结果。因此，Env_002 更适合定位为跨任务验证，而不是机制证明。

### 5.3 可能出现的实验解释

如果 LLM-Once 表现很差而 DERES 表现较好，说明自进化闭环有效。如果 Score-Only Evolution 明显弱于 DERES，说明组件级结构化诊断不是装饰，而是提供了更有效的修改方向。如果 w/o Failure Recovery 出现更多 invalid code、duplicate reward 或停滞，说明搜索成功依赖工程化的失败恢复机制。如果 w/o Expert Schema 初始奖励较差但后续可被修复，则说明 Expert Schema 主要提高初始质量和搜索效率，而不是唯一成功来源。

## 6 讨论：如何避免自嗨

本文需要避免三个过度表述。第一，不能说 DERES 已经实现完全自主智能体。更准确的说法是，DERES 是一个 reward evolution agent，其动作空间是奖励代码修订，反馈来自策略训练和外部评价。第二，不能把 RAG 作为主创新。当前版本更适合称为 Expert Schema Prompting，即用固定专家模板和公式算子约束 LLM 的奖励设计过程。第三，不能把一次成功案例包装成稳定结论。1 seed 成功可以帮助发现机制和写 case study，但正式论文必须用 5 seed 防止偶然性。

从 EI 会议论文角度看，本文的优势不在于提出复杂理论，而在于把一个现实问题讲成完整工程闭环：LLM 写出的奖励函数经常失败，但失败不是终点；训练日志可以被组织成诊断证据；Agent 可以基于证据执行局部修复；记忆和 best archive 保证搜索过程可追踪、可恢复。这条故事具有工程可实现性、实验可验证性和一定新颖性。

## 7 结论

本文提出 DERES，一种基于结构化诊断的大语言模型奖励函数自进化搜索框架。与一次性奖励生成或独立多候选采样不同，DERES 将失败奖励函数的训练结果转化为后续奖励修复的结构化证据，并通过 Diagnostic Reflection Agent、Reward Memory、Best Reward Archive 和 Failure Recovery 形成可追踪的自进化闭环。先导实验表明，该框架具备从失败初始奖励中逐步搜索出有效奖励的潜力，并可迁移到连续控制任务。后续工作将通过冻结配置下的多 seed 实验、消融分析和最终策略评估进一步验证方法的稳定性与泛化性。

## 8 后续必须补充的实验与写作任务

### 8.1 实验任务

| 优先级 | 实验 | 目的 |
| --- | --- | --- |
| 最高 | Env_001 DERES 5 seed × 10 iteration | 主结果，证明自进化与救援能力 |
| 最高 | LLM-Once | 证明不是一次生成成功 |
| 最高 | Score-Only Evolution | 证明结构化诊断有用 |
| 最高 | w/o Failure Recovery | 证明失败恢复是关键机制 |
| 高 | Independent-K10 | 证明顺序诊断优于独立采样 |
| 高 | Env_002 final 100 eval | 连续控制泛化验证 |
| 中 | w/o Expert Schema | 验证专家模板主要提升初始质量和效率 |
| 中 | w/o Memory | 验证历史记忆作用 |

### 8.2 图表任务

| 图号 | 内容 | 说明 |
| --- | --- | --- |
| Figure 1 | DERES 框架图 | 强调 Diagnostic Reflection Agent 与 self-evolution loop |
| Figure 2 | Env_001 多 seed 搜索轨迹 | 候选得分 + best-so-far |
| Figure 3 | Initial-to-Best paired improvement | 直接展示失败奖励被修复 |
| Figure 4 | Success@K / FirstSolveIter | 展示搜索效率 |
| Figure 5 | Case study | 展示一个 seed 的诊断—干预—改善链条 |
| Table 1 | 主结果表 | DERES 与 baselines 对比 |
| Table 2 | 消融实验表 | 结构化诊断、失败恢复、Expert Schema |

### 8.3 写作注意事项

正式投稿时应将本文中的“先导实验显示”替换为“实验结果表明”；如果某些结果尚未重跑，则必须保留“preliminary”或“pilot”表述。不要隐藏失败 seed，失败 seed 反而可以服务于 DERES 的故事：该方法不是保证每次成功，而是通过结构化诊断提高失败奖励被修复的概率。
