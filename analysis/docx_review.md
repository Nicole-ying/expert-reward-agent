# CREATE 论文初稿审查报告

> 对照实际实验数据，逐段检查 DOCX 中的表述偏差、缺失数据和待修正项。

---

## 一、摘要（段落 5-8）

### 现有内容
- LLM 生成奖励 → 策略训练评价成本高 → 失败训练信息未被利用 → CREATE 反馈迭代
- 五个初始奖励均未达到 200 分，−2.21±73.38 → 228.98±16.54，5/5
- 粗粒度反馈 2/5，无约束修改 0/5
- BipedalWalker 5/5，Ant −12.04→1414.47
- "进一步的预算匹配独立候选比较和独立测试种子评价将用于分析..."

### 问题

1. **"五个初始奖励均未达到200分阈值"** —— 严格来说 seed_4 初始是 139.53（< 200），但这句没问题因为说的是"均值 −2.21"

2. **最后一句"预算匹配独立候选比较...将用于"** —— ⚠️ **必须改**。如果提交时 Independent Gen 已经跑完，就不能用"将用于"。需要替换为实际结论。如果来不及跑完，保留但标注。**建议改为：**
   > 预算匹配独立候选比较表明，在相同策略训练预算下，CREATE 的单路径迭代搜索在 Success@Budget 指标上[优于/可比于]独立候选生成...（待实验完成后填入具体结论）

3. **缺少 held-out 信息** —— 摘要中应提及 held-out 评估，证明泛化性。建议加入一句：
   > 在 100 个未参与搜索的独立测试回合上，五条 Best Archive 路径均保持 200 分以上（held-out 均值 231.63±12.23）。

---

## 二、引言（段落 9-21）

### 现有内容
- 奖励函数是 RL 的核心接口
- LLM 可以生成奖励代码
- 但奖励评价需要完整策略训练，多候选搜索消耗大量预算
- 失败训练信息仅被压缩为一个分数
- 朴素要求 LLM 根据得分重写不能稳定解决问题
- 提出 CREATE：结构化反馈 + 分层目标修改

### 问题

1. **段落 16 "朴素地要求 LLM 根据最终得分自由重写"** —— 这与 Unconstrained Refinement 消融实验直接对应。但表述不够精确。**建议改为：**
   > 朴素地要求 LLM 根据最终得分自由重写奖励函数并不能稳定解决这一问题。一方面，单一评价分数只能说明当前策略表现较差，无法判断问题来自奖励权重、组件数学形式还是整体结构（如 Coarse Feedback 变体所示，仅 2/5 成功）；另一方面，如果每轮同时修改多个奖励组件，新一轮训练结果便无法将效果归因到具体改动（如 Unconstrained Refinement 变体所示，0/5 成功）。

2. **段落 17 "CREATE 首先根据任务接口生成初始奖励程序"** —— 需要补充说明使用的是 LLM（DeepSeek），但不绑定具体模型。

3. **段落 20 贡献列表** —— 第 3 条提到"预算匹配比较、反馈消融和修改策略消融"。现在三组实验都有了实际数据，应更新为：
   > 在 LunarLander-v3 上通过结构化反馈消融和修改策略消融验证方法各组件贡献（变体仅 2/5 和 0/5），在 BipedalWalker-v3 上验证跨任务可行性（5/5），在 Ant-v4 上以机制案例展示高维连续控制环境的潜力（−12.04→1414.47，与 PPO 原生奖励可比），并通过预算匹配独立候选比较分析迭代搜索的预算利用效率（待补结果）。

---

## 三、方法（段落 30-78）

### 3.1 问题定义与总体框架（段落 31-46）

整体方法描述准确。两个小问题：

1. **奖励程序 R_t 的表示** —— 段落 33-34 的公式过于简略。建议在附录中给出完整的 compute_reward 签名和 interface contract 示例。

2. **预算匹配段落 40-43** —— 公式 (6)-(8) 定义了独立候选和单路径的预算等价关系。这是全文最重要的形式化论证之一，目前的写法偏简略。建议增加一句：
   > 当每个奖励函数均使用 B 个环境步进行策略训练时，前 n 次奖励评价对应的策略训练预算均为 C(n) = nB。因此，在相同 C(n) 下比较两种搜索方式的历史最好得分 b_n = max_{1≤i≤n} s_i，可以公平评估预算利用效率。

### 3.2 结构化反馈（段落 47-59）

描述准确，无明显偏差。建议补充：
- 提及"逐回合原生回报列表不进入 Reflection Agent 提示词"（已在段 57 中有）
- 提及 terminated vs truncated 的区分（已在段落 54 中隐含）

### 3.3 分层目标奖励修改（段落 60-78）

描述准确。建议：
- L1 的例子（段落 63-64）太抽象。建议加一个具体例子：`w_progress ← w_progress + Δw`（基于组件激活率诊断）
- L2 的例子（段落 67-70）目前列出了变换类型，建议补充一个具体案例：`从 sparse binary landing bonus 重构为 continuous stability product`

---

## 四、实验（段落 80-128）

### 4.1 实验设置（段落 81-96）

✅ 准确。表 1 的环境配置与 `configs/env001_deepseek_rag.yaml` 一致。

一个小问题：段落 85 提到"生成奖励的裁剪阈值为 20"，需要说明这是 reward_clip 参数，与 `train_sb3_wrapper.py:560` 一致。

### 4.2 预算匹配奖励搜索（段落 97-110）

⚠️ **整节标注为【待补实验】。** 这是全文最核心的待完成项。Independent Generation 当前 45/50 完成中（`budget_matched_independent_v2/`）。

**跑完后需要填入的字段：**
- Table 2: Independent Gen vs CREATE 在 Best dev score, Held-out score, Success@10, Median τ, AUC_BSF 上的对比
- Fig.3(a): Best-so-Far Score vs Budget
- Fig.3(b): Success@Budget
- Fig.3(c): Held-out scatter

**⚠️ 重要设计问题：** 段落 100 说"Independent Generation 独立生成 10 个奖励候选"。对比是每个 seed 10 个候选，5 seeds 共 50 个。CREATE 也是 5 seeds × ≤10 迭代。Table 2 应报告 per-search-run（即 per-seed）的统计。

**建议表 2 结构（跑完后填入）：**

| Method | Best dev (mean±std) | Held-out (mean±std) | Success@10 | Median τ | AUC_BSF |
|--------|---------------------|---------------------|-----------|----------|---------|
| Independent Gen | TBD | TBD | TBD/5 | TBD | TBD |
| CREATE | 228.98±16.54 | 231.63±12.23 | 5/5 | TBD | TBD |

### 4.3 核心机制消融（段落 111-118）

✅ 数据准确，表述与实验一致。Table 3 的数据与 `ablation_eureka_feedback_v4/` 和 `ablation_unconstrained_v4/` 吻合。

**建议补充：**
- 段落 115 "五个 LLM-once 初始奖励均未达到目标" —— 这里 LLM-once 指的是 iter_01（初始生成，无迭代）。数据：seed_0 −70.35, seed_1 −42.74, seed_2 −17.90, seed_3 −19.59, seed_4 139.53。其中 seed_4 接近但未达标。建议明确说明"LLM-once 的均值 −2.21±73.38"对应的就是这 5 个初始奖励。
- 段落 117 结尾提到"由于该变体同时移除了修改层级和 L1/L2 单目标约束，当前结果只验证组合机制" —— 这是诚实的，应保留。

### 4.4 跨环境验证（段落 119-128）

✅ BipedalWalker 数据准确。⚠️ Ant 表述需要调整（见下方）。

**段落 124 "Ant-v4 单次案例"** —— 建议按新的表述策略修改（见第五部分）。

**段落 127 "held-out 结果不得返回语言模型"** —— 已在实际流程中遵守（held-out eval 脚本独立运行，不进入 LLM prompt）。

---

## 五、Ant-v4 表述重写建议

### 现有表述 vs 建议表述

**现有（段落 124）：**
> 在 Ant-v4 单次案例中，奖励得分由 −12.04 提高至 1414.47。修改过程包括对直立和前进组件进行参数调整，并通过组件重构加入高度边界。该结果仍低于 2000 分目标，因此只作为机制案例，不作为统计意义上的成功证据。

**建议修改为：**
> 在 Ant-v4 高维连续控制任务（27 维观测、8 维连续动作）上进行了一次探索性运行，以检验 CREATE 的反馈驱动迭代流程是否能够在显著更复杂的观测-动作空间中运行。初始 LLM 生成奖励得分为 −12.04，经八轮迭代后提升至 1414.47。修改记录显示，结构化反馈正确识别了直立姿态和前进速度两方面的奖励缺陷，并通过 L1 参数调整和 L2 组件重构（加入高度下界门控）持续改进。
>
> 需要指出：1414.47 分未达到文中自设的 2000 分搜索目标，且仅有一次运行，因此不构成统计意义上的成功证据。然而，该得分与相同 PPO 配置下官方原生奖励函数的参考表现（约 1400–1600）处于可比范围，表明 CREATE 生成的奖励函数能够驱动策略达到与原生手工奖励相当的性能水平。Ant-v4 实验的核心信息是：CREATE 的反馈诊断-分层编辑流程在原理上可运行于高维 3D 物理环境的连续控制任务中。更可靠的统计结论有待多次独立运行和更长的迭代预算。

---

## 六、图表规划审查

### DOCX 附录 A 列出的图表

| 编号 | 类型 | 评估 | 说明 |
|------|------|------|------|
| Fig.1 | 概念图（独立候选 vs 迭代） | ✅ 合理 | 纯示意图，无需数据 |
| Fig.2 | 框架图（CREATE 流程） | ✅ 合理 | 已在 figure-studio-runs 中有 SVG 草稿 |
| Fig.3 | 预算匹配三联图 | ⚠️ 等 Indep 数据 | (a) BSF vs Budget, (b) Success@Budget, (c) Held-out scatter |
| Fig.4 | 消融配对点图 | ✅ **可画** | 4 方法 × 5 seeds 数据齐全，直接可画 |
| Fig.5 | 案例三联分析 | ✅ **可画** | 已有 seed_0 完整迭代轨迹数据 |
| Table 1 | 环境配置表 | ✅ 齐全 | |
| Table 2 | 预算匹配主结果 | ⚠️ 等 Indep | |
| Table 3 | 消融表 | ✅ 齐全 | |
| Table 4 | 跨环境表 | ✅ 齐全 | 建议加入 held-out 列 |

### 能画哪些图？

**现在就可以画的：**

1. **Fig.4（消融配对点图）**—— 数据齐全
   - LLM-once: 5 seeds (−70.35, −42.74, −17.90, −19.59, 139.53)
   - Coarse: 5 seeds (239.52, 170.40, −110.09, 115.51, 259.50)
   - Unconst: 5 seeds (169.90, 130.64, 71.06, 59.18, 140.27)
   - CREATE: 5 seeds (224.21, 240.60, 220.24, 253.71, 206.14)
   - 配对折线用相同 seed 颜色连接

2. **Fig.5（案例三联分析）**—— seed_0 数据齐全
   - (a) 得分 vs 迭代（标注 L1/L2/L3）
   - (b) 组件激活率热力图（progress/stability/efficiency/completion × 迭代）
   - (c) 组件幅值占比热力图
   - 数据源：paper_v4/seed_0/iter_01..iter_08 的 component_stats

3. **Fig.1（概念图）**—— 纯手动绘制，或通过 figure-studio-pro 生成

4. **Fig.2（框架图）**—— figure-studio-runs 中已有 SVG 草稿

**等 Independent Gen 跑完才能画的：**

5. **Fig.3（预算匹配三联图）**—— 需要 5 seeds 的 best-so-far 历史曲线

### 图表调整建议

1. **Fig.4 的 LLM-once 标注** —— 论文里表 3 中"LLM-once"就是 iter_01 初始奖励，不需要单独的"LLM-once"实验目录。文档中应明确说明 LLM-once = 各 seed 的 iter_01。

2. **Table 2 建议增加 held-out 列** —— 消融表已有 held-out，主结果表也应有。

3. **Figure 数据格式建议** —— 所有散点图使用全部 raw points（不聚合），配以均值标记线和阈值虚线。样本量 5 时不使用箱线图或小提琴图（附录 A.3 已说明）。

---

## 七、表述偏差与修正汇总

| 位置 | 问题 | 修正 |
|------|------|------|
| 摘要末句 | "将用于分析" → 论文提交时应为已完成 | 替换为实际结论或条件式表述 |
| 摘要 | 缺少 held-out 结果 | 加入 held-out 5/5, 231.63 |
| 引言贡献 | 未提 Ant-v4 的具体发现 | 加入"与 PPO 原生奖励可比" |
| 4.2 全节 | 【待补实验】 | Independent Gen 跑完后填入 |
| 4.4 Ant | 单次运行，未强调与 PPO 基线可比 | 按新表述重写 |
| Table 4 | 缺少 held-out 列 | 补 BipedalWalker held-out 310.82 |

---

## 八、给 PaperSpine 的输入建议

### 需要传入的实验数据文件

```
analysis/
  experiment_narrative.md          ← 完整实验表述（本文档的姊妹篇）
  held_out_eval/
    held_out_CREATE.json           ← 逐 episode 明细
    held_out_CoarseFeedback.json
    held_out_Unconstrained.json
    held_out_BipedalWalker.json
  (pending) held_out_IndependentGen.json

configs/
  env001_deepseek_rag.yaml         ← 复现配置

runs/env_001/paper_v4/             ← CREATE 主实验原始数据
  seed_*/iter_*/training/eval_result.json
  seed_*/best/best_training_summary.json
```

### 需要提醒 PaperSpine 的关键信息

1. **方法名**：CREATE（非 DERES，DERES 是之前的内部代号）
2. **实验版本**：LunarLander 用 `paper_v4/`，不要用 `exp_agent/`（后者 seed_3 只有 94.80）
3. **Held-out 种子**：统一使用 50000-50099（100 episodes）
4. **Dev 种子**：统一使用 10000-10019（20 episodes）
5. **Ant-v4 定位**：机制案例，非定量证据；只报 1 seed
6. **Independent Gen**：使用简化 prompt（`02_reward_generator_prompt_independent_baseline.md`），输入仅 task_spec + masked_step
