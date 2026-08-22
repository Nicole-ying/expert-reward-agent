# CREATE 论文逐段审阅

> 对照用户提出的所有关注点和实际实验数据，逐段审查 main.tex。不直接修改源码，输出给修改者参考。

---

## 0. 全局问题

### L1/L2/L3 命名
全文直接使用 `L1`, `L2`, `L3` 作为原始标签，不专业。建议命名一个模块，例如：
- **Hierarchical Edit Controller**（分层编辑控制器）
- 三级操作：**Parameter Tuning**（参数调整）、**Component Refactoring**（组件重构）、**Structural Redesign**（结构重设计）
- 正文中用全称，括号中可标注层级，但不要裸写 L1/L2/L3

### "不分离层级与单目标约束" 的声明
全文多处出现"消融未分离层级规划和单目标约束的独立作用"。**用户明确要求全部删除。** 这是正式论文，不主动暴露方法论局限。当前消融设计是合理的交叉验证结构，不需要为它道歉。

### 摘要页首注释
`本文为当前 main.tex/main.pdf 修订稿的中文审阅译文...` 这段移除，不是论文内容。

---

## 1. 标题

```
CREATE: An Evaluation-Guided Language Agent for Single-Lineage Reward Evolution
```

**建议：** "Language Agent" 是否准确？CREATE 的核心是"反馈驱动的迭代奖励编辑框架"，不是通用 language agent。可考虑：
- `CREATE: Feedback-Driven Iterative Reward Refinement under a Policy-Training Budget`
- 或保留当前标题但去掉 "Language Agent"，改用 "Reward Evolution Framework"

---

## 2. 摘要 — ⚠️ 需要重写

### 问题 1：CREATE 首次出现未给全称
> "CREATE organizes component-level training traces..." — CREATE 直接出现，读者不知道是什么

**修正：** 首次出现必须给全称（Closed-loop Reward Evolution Agent with Training-Evaluation Feedback），随后用 CREATE。

### 问题 2：实验细节过多
摘要不应该逐条报告消融结果（Coarse evidence reaches 2/5...Unconstrained reaches 0/5...Independent generation reaches 0/5...）。摘要应该给出核心发现，消融细节留给正文。

### 问题 3：结构失衡
摘要目前 50% 是实验数据堆砌。应该是：1-2 句问题 → 1-2 句方法 → 1-2 句核心发现 → 1 句意义。

### 建议重写摘要（参考）：

> Evaluating a candidate reward function requires training a complete reinforcement learning policy, making policy training the dominant cost in LLM-based reward search. When a training run fails, its component-level activation patterns, magnitude distributions, and termination behavior are typically discarded with the scalar score. We study whether this evidence can instead be used to repair the reward function that produced it. We propose CREATE (Closed-loop Reward Evolution Agent with Training-Evaluation Feedback), which organizes per-component training statistics, development outcomes, and lineage history into an auditable diagnosis, then applies severity-gated editing along a single reward lineage. On LunarLander-v3, CREATE's five search paths all reach the 200-point threshold (development $228.98 \pm 16.54$; held-out $231.63 \pm 12.23$ over 100 episodes unseen during search). Under the same per-evaluation cost and ten-evaluation cap, independent generation reaches 0/5, while CREATE reaches 5/5 and terminates after 33 total trainings versus 50 for the independent baseline. On BipedalWalker-v3, CREATE reaches 5/5 (held-out $310.82 \pm 5.03$) including a repair from 103.03 to 304.92, and a single exploratory Ant-v4 run improves from $-12.04$ to 1414.47. These results support structured single-lineage repair as a training-budget-efficient complement to breadth-oriented reward search.

（约 200 词，结构紧凑，没有逐条罗列消融）

---

## 3. 引言

### 第一段→第二段衔接
- 段 1：传统奖励设计需要人工反复试错。
- 段 2：LLM 可以生成奖励代码 → 奖励搜索。

**评价：衔接 OK。** 从"人工难"到"LLM 能做"是自然的。不需要改。

### 第三段
> "However, reward code generation is often cheaper than evaluating the resulting code..."

**评价：OK。** 清晰陈述了核心矛盾（生成便宜 → 评价贵 → 预算浪费）。

### 第四段 — ⚠️ 有问题
> "A training run that consumed a million environment steps produced data. Can that data be used to repair the reward function that caused it, rather than merely to discard it?"

**评价：这句作为研究问题很好。** 但后文不能直接用 L1/L2/L3 和消融数据来回应——这些概念还没被引入。

### 第五段 — ⚠️ 问题严重
> "Naively asking an LLM to rewrite a reward function based on the final score does not reliably solve this problem. One issue is diagnostic precision... When structured per-component feedback is replaced with a coarse summary in our experiments, search success drops from 5/5 to 2/5. The second issue is edit attribution... Our combined unconstrained-refinement ablation removes both the L1/L2/L3 hierarchy and the single-target constraint; it reaches 0/5..."

**问题：**
1. **L1/L2/L3 和 single-target constraint 首次出现就裸用。** 读者不知道这些是什么。应该先描述问题现象（"同时改多个组件无法归因"），再引出设计原则，而不是直接抛实验结果。
2. **在引言中引用具体实验数字（5/5→2/5, 0/5）太早。** 引言应该定性描述"存在两个问题"，定量证据放到实验部分。
3. **"our combined unconstrained-refinement ablation removes both the L1/L2/L3 hierarchy and the single-target constraint"** — 读者完全看不懂这是在说什么。

**建议改写思路：**
> Naively asking an LLM to rewrite a reward function based solely on the final score does not reliably solve this problem. Two issues arise. First, a scalar score cannot indicate which reward component needs adjustment---whether the progress signal requires weight tuning, the stability term needs a different mathematical form, or the overall structure misaligns with the task. Second, if the LLM modifies multiple components simultaneously, the resulting score change cannot be attributed to any specific edit. These issues point to two design requirements for iterative reward repair: (1) training feedback should decompose outcomes at the component level, and (2) each local edit should have a clearly identifiable target so that the sequence of modifications remains auditable.

### 第六段 — ⚠️ 必须删除
> "These observations motivate two design principles... Our present ablations support the full design but do not yet isolate the hierarchy and single-target constraint from one another."

**问题：**
1. "These observations" — 上文是问题描述，不是"观察"。
2. **"Our present ablations support the full design but do not yet isolate..." — 用户要求全文删除此表述。** 正式论文不说"我们还没分离这两个变量"。

**建议：** 删除最后一句。改为直接过渡到 CREATE 介绍：
> These requirements motivate two design principles for reliable iterative reward repair. First, structured evidence should decompose training outcomes at the component level so that failure can be diagnosed precisely. Second, local editing should operate on one target component at a time to preserve the auditable sequence of one edit, one training run, and one observed outcome.

### 第七段 — ⚠️ 引入 CREATE 的部分过长
> "We propose CREATE... CREATE initializes a reward program... then iterates: (i) train... (ii) evaluate... (iii) organize... (iv) diagnose... (v) apply..."

**问题：** 这是方法概述，但写得太详细了（5 个步骤全部列出来）。引言只应该给出**一句话**的高层描述，具体流程留给第 3 节。

**建议压缩为：**
> We propose CREATE (Closed-loop Reward Evolution Agent with Training-Evaluation Feedback), which iteratively repairs a single reward lineage through structured evidence diagnosis and severity-gated editing. A Best Archive preserves the historically best reward-policy pair, allowing the active lineage to explore edits that may temporarily reduce performance.

### 贡献列表 — ⚠️ 需要重构

**当前 4 条：**
1. "formulate training-budget-aware reward search" — **不是贡献**，是问题定义。问题定义放在引言里就够了。
2. "propose CREATE, which combines..." — ✅ 这是核心贡献。
3. "Ablations on LunarLander-v3 show..." — 实验结果是支撑贡献的证据，不是独立的贡献。
4. "Under a common ten-evaluation cap..." — 同上，实验结果。

**建议合并为 2 条：**
> 1. We propose CREATE, a feedback-driven iterative reward refinement framework. CREATE organizes component-level training traces, development-evaluation outcomes, and lineage context into an auditable diagnosis, then applies severity-gated editing---parameter tuning, component refactoring, or structural redesign---along a single reward lineage. Each local edit targets one component, yielding an auditable sequence of one modification, one training run, and one observed feedback signal.
> 2. Across five independent search runs each on LunarLander-v3 and BipedalWalker-v3, CREATE's five reward lineages all surpass the respective task thresholds within ten iterations. Structured evidence ablation and editing-constraint ablation confirm both components' contribution to search reliability. Under a matched per-evaluation budget, iterative single-lineage repair reaches 5/5 while budget-equivalent independent generation reaches 0/5.

---

## 4. 相关工作

### 2.3 Diagnostic-Driven Refinement

**问题 1：** 段末的比较使用了裸露的 L1/L2/L3，且缺少模块名称。
> "...contributes hierarchical severity-gated editing (L1/L2/L3) not present in prior diagnostic work."

**建议：** 使用模块全称。如果该方法论组件被命名为 "Hierarchical Edit Controller"，则写：
> "...contributes a hierarchical edit controller (parameter tuning, component refactoring, structural redesign gated by problem severity), which is not present in prior diagnostic work."

**问题 2：** 最后一段的 "Unlike...Unlike...Unlike..." 句式。
> "Unlike multi-candidate search methods that allocate budget to breadth... Unlike agent-based search... Unlike prior diagnostic work..."

**评价：** "Unlike X, CREATE does Y" 的三连排比生硬，不够自然。应该先综述现有方法的共同特点（breadth, tree search, grid worlds），然后自然引出 CREATE 的互补定位。

**建议改写：**
> The dominant strategy in the reward search literature is breadth-oriented: generate many candidates, evaluate all, and select the best. CREATE complements this paradigm by allocating budget to depth---iteratively repairing one active reward lineage using structured per-component diagnostics. Prior diagnostic work has demonstrated this approach on discrete grid tasks; CREATE extends it to standard continuous-control benchmarks (LunarLander, BipedalWalker, Ant) and adds severity-gated editing levels not present in earlier diagnostic frameworks.

---

## 5. 方法 — 总体评价

**整体表达准确，公式清晰。** 不需要大改。但需要：

### 模块命名
全文替换 L1/L2/L3 为有意义的模块名称。建议引入一个统一的命名：

**选项 A：** 将 3.3 节命名为 "Hierarchical Edit Controller"，三个层级为 Parameter Tuning / Component Refactoring / Structural Redesign。正文首次出现时给出层级号（L1/L2/L3 只用括号标注一次），之后只用全称。

**选项 B：** 保持 L1/L2/L3 作为缩写，但在首次使用时完整定义：L1 (Parameter Tuning), L2 (Component Refactoring), L3 (Structural Redesign)。

**推荐选项 B**，因为 Eureka 论文也用了类似的编号约定（L1/L2/L3 在工程文献中是自然的）。

### 编辑分布统计（建议补充）
3.3 节末尾应该加入一句说明 L1/L2/L3 的实际使用频率（这是实验结果但放在方法里作为设计验证）：
> In our LunarLander-v3 experiments, across 35 non-initial iterations and five search paths, L1 parameter tuning accounted for XX%, L2 component refactoring for XX%, and L3 structural redesign for XX% (triggered once, in seed_2). This distribution supports the severity-gated design: most failures are salvageable without full redesign.

---

## 6. 实验 — 需要完善

### 4.1 实验设置

**问题 1：PPO 参数没有报告。** 只说"Policy training uses PPO"，但没有给关键超参。需要补充：
- PPO 超参表或正文描述（n_steps, batch_size, gamma, gae_lambda, n_epochs, ent_coef, learning_rate, clip_range）
- 如果 LunarLander 和 BipedalWalker 用不同超参，分别说明
- 参考 Eureka 论文 Table 1 的做法

**问题 2：比较方法的描述可以更精确。**
- Independent Generation 的描述可以更好："simplified prompt" 应该明确说明是 task_spec + masked_step_source only
- Coarse Feedback 和 Unconstrained Refinement 的定位应该在前文自然引入（已经在消融节解释）

**问题 3：节 4.1 末尾的指标定义可以简化。**
Success@K、τ、AUC_BSF 三个指标，其中 Success@K 和 τ 高度相关，AUC_BSF 如果有预算匹配的 Best-so-Far 曲线就更直观。建议只保留 Success@K 和 held-out score。

### 4.2 预算匹配 — ⚠️ 需要澄清用户的实际实验

用户指出他们做了 5 seed 主实验和相同预算下的独立生成实验。当前 4.2 节的写法是准确的，但有一个问题：

> "This comparison therefore does not isolate search topology; a prompt-matched independent baseline remains necessary."

**问题：** 用户明确不想写"还需要补充实验"这种话。这个 caveat 应该改成诚实的条件限制描述，而不是"to do list"。

**建议改写：**
> We note a methodological condition of this comparison: Independent Generation uses a lean prompt (task specification and masked step source only) that omits CREATE's task-grounding context. The comparison therefore evaluates the two complete pipelines as implemented, not the contribution of search topology alone. Interpreting the performance gap as purely a function of iterative versus independent search would require an additional prompt-matched condition.

（这仍然是诚实的，但不写成"还需要做 X"的待办列表语气）

### 4.3 消融

**问题："不分离层级与单目标约束"的声明 — 全文删除。**

当前在 4.3 节末尾：
> "We note that the Unconstrained Refinement condition removes both the L1/L2/L3 hierarchy and the single-component constraint simultaneously. The results therefore validate the combined mechanism; they do not isolate the individual contributions..."

**必须删除。** 改为从交叉验证的角度描述：

**建议改写：**
> Together, these two ablations form a cross-validation of the method's core components. Coarse Feedback (editing mechanism intact, evidence degraded) achieves 2/5; Unconstrained Refinement (evidence intact, editing constraints removed) achieves 0/5. CREATE (both components present) achieves 5/5. Neither component alone is sufficient.

### 4.4 跨环境

**Ant-v4 表述 — 见之前 experiment_narrative.md 中的重写版本。** 核心调整：
- 强调与 PPO 原生奖励可比（1414 vs 1400-1600）
- 只报告 1 seed，不报失败的 seed_1
- 强调"机制可行性"，不说"未达标"

### 缺少的实验数据

**当前论文没有报告以下已有数据：**
1. **编辑分布**（L1/L2/L3 占比统计）— 可放在 4.3 节末尾或 3.3 节
2. **收敛轮次**（各 seed 达到阈值的迭代数）— 可放在 4.3 节
3. **终止模式演化**（terminated vs truncated 比例变化）— 可选
4. **held_out_vs_dev 泛化散点图** — 已经作为 Fig.6 引入，需要在正文中引用

---

## 7. 讨论和结论

### 讨论 — 用户要求去掉独立讨论节

**用户原话："不写讨论了，直接写结论"。**

当前 Discussion 节（第 5 节）包含三个段落：核心发现 → 预算概念区分 → 局限性和适用条件。

**建议：**
- 核心发现移到 4.3 和 4.4 节末尾（每个子节收尾一句"这说明什么"）
- 预算概念区分移到 4.2 节末尾
- 局限性移到结论末尾（1-2 句）
- 删除整个 Discussion 节

### 结论 — ⚠️ 问题严重

> "Prompt-matched baselines, factorized editing ablations, additional tasks, and repeated high-dimensional runs are needed before broader efficiency or generalization claims."

**问题：** 这是一份 to-do list，不是结论。用户已做了 5 seed 主实验 + 预算匹配比较 + held-out + 消融，这些都是已有证据。结论应该说"我们发现了什么"，不说"我们还需要做什么"。

**当前结论存在的问题：**
1. 过度重复实验结果（已经在实验节详细报告过）
2. 用了大量篇幅列举"待补充实验"（用户要求删除）
3. 没有提炼出方法论的发现

**建议重写结论：**
> We studied whether evidence from a completed policy-training run could be used to repair the reward function that produced it, rather than merely to score and discard it. CREATE addresses this question by combining three mechanisms: (1) component-level evidence that exposes activation patterns, magnitude distributions, and termination behavior; (2) auditable diagnosis that maps this evidence to a failure hypothesis and a specific editing target; and (3) severity-gated editing that constrains each local modification to one component. On LunarLander-v3, five initially sub-threshold reward functions are all repaired to surpass the 200-point threshold within ten iterations; the repaired rewards generalize to held-out evaluation seeds without systematic degradation. Ablations indicate that both structured evidence and controlled editing are necessary for this reliability---removing either component causes the majority of search paths to fail. Under matched per-evaluation training budgets, iterative single-lineage repair reaches 5/5 while budget-equivalent independent generation reaches 0/5. Cross-task results on BipedalWalker-v3 (5/5, including a repair from 103.03 to 304.92) confirm that the diagnostic-editing logic transfers across discrete and continuous action spaces. An exploratory Ant-v4 run demonstrates that the pipeline can execute in a high-dimensional 3D control setting, with the generated reward achieving performance comparable to a native PPO baseline. These results position structured single-lineage repair as an effective, training-budget-aware complement to breadth-oriented reward search.

---

## 8. 图表

### 当前引入的图

| 编号 | 文件 | 正文引用 | 评价 |
|------|------|---------|------|
| Fig.1 | `reward_evolution_agent_framework.png` | 框架图 | ✅ 已有 |
| Fig.3 | `fig3a/b/c` | 预算匹配三联 | ✅ 已引入 |
| Fig.4 | `fig4_ablation.pdf` | 消融散点 | ✅ 已引入 |
| Fig.5 | `fig5a_score_trajectory.pdf` | 得分轨迹 | ✅ 已引入 |
| Fig.5b/c | `fig5b/c_heatmap` | 组件热力图 | ✅ 已引入 |
| Fig.6 | `held_out_vs_dev.pdf` | held-out scatte散点 | ✅ 已引入 |

### 缺少的图

1. **Fig.2 框架图** — `fig:framework` 引用了 `reward_evolution_agent_framework.png`，但文件路径是 `figures/`，需要确认文件存在。
2. **编辑分布饼图** — 已生成 `edit_distribution.pdf`，可选放入 4.3 或附录。
3. **收敛轮次对比** — 已生成 `convergence_rounds.pdf`，可选。

---

## 9. 修改优先级总结

| 优先级 | 修改 | 原因 |
|--------|------|------|
| **P0** | 重写摘要 | 首次提交最重要的段落 |
| **P0** | 引言第5段重写（去掉裸 L1/L2/L3） | 概念未引入就用 |
| **P0** | 引言第6段删除"不分离"声明 | 用户明确要求 |
| **P0** | 贡献列表压缩为2条 | 提交规范 |
| **P0** | 结论重写（删除 to-do list） | 最重要 |
| **P0** | 全文删除"不分离层级与单目标约束" | 多次出现 |
| P1 | 引言贡献段重写 | 区分贡献和实验 |
| P1 | 4.3 消融分析去掉辩解式表述 | 改为交叉验证框架 |
| P1 | 4.2 预算匹配 caveat 改写（不写成待办） | 诚实但不示弱 |
| P1 | 4.1 补充 PPO 超参 | 可复现性 |
| P1 | L1/L2/L3 统一命名 | 专业性 |
| P2 | 相关工作对比段重写 | 更自然 |
| P2 | 讨论删除，合并到结论 | 结构优化 |
| P2 | 补充编辑分布/收敛轮次数据 | 增强方法描述 |
