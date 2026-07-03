# 深度诊断奖励自进化：修改与实验计划

## 当前实现状态

- 已接入：任务无关初始生成 Prompt、主要任务路由、任务路由专家骨架检索。
- 已接入：训练组件 mean/abs_mean/active-rate/active-mean 和完整训练回合累计统计。
- 已接入：固定 seed 外部评估、terminated/truncated、逐回合组件与行为事实。
- 已接入：Lander 与 Bipedal observation-only 诊断适配器；无法确认的终止原因保持 unknown。
- 保持不变：Reflection Agent 的 system prompt、user prompt 组装和纯表格 Agent Memory 与 exp_v6 一致；改进只通过增强 `training_feedback.md` 提供更多证据。
- 已接入：全局最佳保留、完全重复候选免训练、证据驱动 Fresh restart。
- 已接入：初始奖励2-4个直接 reward terms 的组件预算；迭代 Agent 本身不增加新的组件数量硬约束。
- 正式主实验目录：`paper_main/seed_N/iter_XX`，不沿用开发期 v6/v7 命名。
- 尚未执行：昂贵的 Lander/Bipedal 正式多 seed、组件消融和最终100回合复评。

## 1. 研究命题

将失败候选从种群搜索中的淘汰样本，转化为可继续利用的诊断证据。系统执行：观察训练行为、提出可证伪假设、实施局部干预、验证预测、保留或回退；停滞时允许结构重建。

核心比较不是“LLM 能否生成好奖励”，而是：在相同完整策略训练预算下，诊断式迭代能否比独立生成取得更高成功率、更少候选数和更清楚的改进归因。

## 2. 当前证据与限制

- exp_v6 当前 7 个 Lander seed 中 5 个达到 200 分，说明从失败父代迭代到有效奖励具有重复可行性。
- seed5 展示了清楚的修复链：连续化稀疏 proxy，再将全局稳定约束改为距离条件约束，Iter3 达到 242.41。
- seed0 最佳 152.38，长期停留在 progress + proxy + stability 家族，暴露局部停滞、重启重复和反馈辨识不足。
- seed6 Iter3 正确读取了组件统计，却把高 proxy 激活率直接解释为悬停利用；收紧阈值后激活率未下降且得分下降，说明组件统计不是行为事实或因果证据。
- 当前结果是方法开发证据，不作为修改诊断协议后的最终公平对比结果。

## 3. Prompt 与专家知识

### 3.1 必须修改的偏置

删除顶层 Prompt 中“通常选择 progress_delta_reward”以及固定“主信号 + stability + proxy”组合。顶层 Prompt 只保留任务无关原则：信号可用性、稠密性、尺度、冲突、阶段条件、可利用风险和最小设计。

### 3.2 保留的专家知识

保留骨架库和按任务路由检索。常见骨架是合理先验，不要求不同环境必须产生不同骨架。Lander 与 Bipedal 应由环境接口和任务路由检索到不同候选，而不是由顶层 Prompt 预指定。

### 3.3 Fresh restart

- 不强制每次重启必须换骨架；同一骨架可能仍有可修复空间。
- 将重启历史、最佳表现和已失败干预提供给 LLM，由其选择继续修复或重建。
- 程序计算代码哈希、组件集合和表达式摘要，只作重复告警和实验记录。
- 完全相同代码直接拒绝；近似重复允许，但 LLM 必须说明新证据、新假设和预期差异。
- 若同类干预连续两次未达到预测效果，则提示结构重建，不机械强制。

## 4. 反馈系统

### 4.1 通用层

所有环境统一记录：return、episode length、terminated、truncated、reward error、每组件 mean/abs_mean/min/max/nonzero_rate、激活时条件均值、每回合组件累计值、action 绝对均值和变化量、observation 各维统计。

`original_env_reward` 仅用于外部评估，不放入训练组件比率，也不向生成模型暗示它参与训练。

### 4.2 任务诊断适配器

适配器只观察轨迹，不生成或修改奖励：

- Lander：最终目标距离、速度、角度、接触持续时间、动作使用，以及 success/failure/timeout/unknown。
- Bipedal：最终水平进度、平均前进速度、hull 姿态、左右接触统计、动作强度，以及 reached-end/fall/timeout/unknown。

`terminated` 本身不等于成功或失败。无法可靠分类时必须输出 unknown。

### 4.3 组件语义

区分 reward term 和 modulator。`dist_gate` 只作为 stability term 的条件变量记录，不能作为独立奖励贡献参与 ratio。

不再使用 signed progress mean 作为唯一分母。并列提供绝对贡献占比、每回合累计贡献、激活率和激活条件均值，由 LLM 综合判断。

## 5. 假设、干预与验证

每轮反思必须输出结构化记录：

1. 观察事实，不包含未经验证的行为解释。
2. 至少一个候选根因及其证据强度。
3. 本轮选择的假设。
4. 单一局部干预；rebuild 可作为大范围例外。
5. 对可观测指标的方向性预测。
6. 训练后的预测验证：supported、refuted 或 inconclusive。
7. 下一步：保留、回退、继续验证或重建。

允许迭代得分下降。系统保留全局最佳奖励，并把被否定的假设写入 memory，避免重复执行同类无效干预。

## 6. 评估协议

- 搜索过程中使用 20 个固定评估环境 seed。
- 最终 Lander 策略使用 100 个固定评估 seed；Bipedal 使用 50，资源允许时提升到 100。
- 父代和子代使用相同评估 seed，进行逐回合 paired comparison。
- 每个正式方法至少使用 5 个独立训练 seed。
- 报告 mean、standard deviation、median、95% confidence interval、成功率和终止类型比例。
- 区分训练随机性与评估随机性。关键父子修改至少补做 3 个训练重复，不能仅凭单次训练声称因果关系。

## 7. Lander 正式实验

在修改后的统一协议下重新运行，不把 exp_v6 与新协议结果直接合并。

### 7.1 主方法与基线

在相同最多 10 次完整策略训练预算下比较：

1. One-shot：一次生成。
2. Best-of-N：独立生成 N 个候选并取最佳。
3. Score-only iteration：只有外部分数和长度。
4. Component iteration：增加组件统计，但无行为诊断和预测验证。
5. Full method：组件、行为、memory、假设验证和停滞重建。

如预算不足，优先保留 Best-of-N、Score-only 和 Full method；One-shot 可由 Best-of-N 的第一个候选得到。

### 7.2 必要消融

- Full method 去掉行为指标，检验 seed6 类误诊是否增加。
- Full method 去掉 memory，检验无效干预重复率。
- 专家知识 on/off 或 full/abstract-principles-only，检验知识是否提高成功率以及是否增加结构同质化。

### 7.3 seed0 专项分析

从 Iter8 最佳奖励建立诊断分叉，仅作为机制分析：继续局部调节、条件化约束、替代 proxy、结构重建。使用相同训练和评估 seeds，判断它是可修复局部最优还是骨架上限。该实验不能替代正式独立 seed 比较。

## 8. Bipedal 迁移实验

- 复用完全相同的搜索控制、memory、假设验证和通用统计代码。
- 只更换环境卡、任务路由知识和 Bipedal 诊断适配器。
- 不把 Lander 的距离、落地或引擎术语写入通用 Prompt。
- 最多 10 轮、5 个训练 seed；先做 1 个 seed 的端到端冒烟实验，再启动正式组。
- 使用官方外部奖励只做评估，生成模型不可读取其公式。
- 记录达到预定阈值所需候选数、训练步数和 wall-clock/GPU time。

## 9. 主要论文指标

- solve rate：在预算内达到环境阈值的 seed 比例。
- candidates-to-solve：首次达标前完整训练的候选数。
- best external return：最终最佳外部性能。
- repair rate：失败父代经后代修复达标的比例。
- useful intervention rate：预测得到支持且性能不退化的干预比例。
- diagnostic calibration：不同置信度下假设被支持的频率。
- duplicate rate：完全重复和近似重复奖励比例。
- search cost：完整策略训练次数、环境步数、LLM 调用、wall-clock/GPU time。

## 10. 理想但可信的结果

- Full method 在 Lander 和 Bipedal 均达到至少 4/5 seed 成功。
- Full method 相比 Best-of-N 和 Score-only 使用更少候选达到阈值，或在相同预算下成功率更高。
- 加入行为诊断后，错误语义推断和被否定干预比例下降。
- memory 降低重复无效干预；停滞重建能修复部分 seed0 类案例。
- 专家知识提高早期候选质量，但不会成为成功的唯一来源；无知识条件仍能通过反馈迭代取得明显改善。
- 搜索曲线允许非单调，最终结论依赖成功率、样本效率和最佳保留，而不是要求每轮都提升。

## 11. 实施顺序

1. 修复评估记录：终止信息、逐回合轨迹统计、paired seeds 和置信区间。
2. 增加通用轨迹统计与 Lander/Bipedal 诊断适配器。
3. 修改组件统计，区分 reward term 与 modulator，增加每回合和条件统计。
4. 改造反思输出为“事实-假设-干预-预测-验证”。
5. 修复 memory 和 Fresh restart：保持全局 best，记录重复但不强制换骨架。
6. 去除顶层 Prompt 的 Lander 结构偏置，保留任务路由专家知识。
7. 用已有 seed5/6 记录做离线回放测试，确认新反馈能识别预测失败。
8. Lander 单 seed 冒烟测试，再做正式多 seed 和消融。
9. Bipedal 单 seed 接管与冒烟测试，再做正式多 seed。
10. 汇总统计、迭代案例、成本曲线和论文图表。

## 12. 论文图表

- 主结果表：两个环境上各方法的成功率、最佳分数、候选数和成本。
- 搜索效率图：横轴完整策略训练次数，纵轴 best-so-far external return。
- 机制图：事实、假设、干预、预测、验证、memory/rebuild 闭环。
- 案例图：seed5 成功修复、seed6 预测被否定、seed0 停滞与重建。
- 消融图：去掉行为反馈、memory、专家知识后的变化。
- 结构图：不同 seed 的奖励家族迁移和重复率，而不是仅展示最终奖励代码。
