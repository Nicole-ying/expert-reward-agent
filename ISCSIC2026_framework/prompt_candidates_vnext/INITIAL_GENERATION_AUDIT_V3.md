# Env-001 vNext 第三次 A/B 初始生成审计

## 协议

- 运行目录：`runs/vnext_initial_ab/env001_prompt_quality_20260801_v3/`
- 调用：1 次 Environment Card；A 为 card-only；B 为 card + paper-v4 Historical Expert Context。
- Prompt tokens：A 约 5203，B 约 6643；Expert Context 增加约 27.7%。

## 环境卡

优点：任务描述没有重复整份 YAML；observation/action 使用表格；`TimeLimit`、合法 `info` 访问和 `original_reward` 禁令正确；明确把精确成功阈值标为未知。

问题：卡片一方面承认阈值未知，另一方面又把 success/failure 写成可可靠识别，前后矛盾。观察到相关状态量只能支持启发式完成质量判断，不能证明精确终局分类。卡片也遗漏了输入契约中已有的 ±20 runtime reward clip，导致两份 Reward Generator 都声称不知道 clip。

## A：card-only

- 组件：`progress`、`terminal`、`energy`。
- 自动验证失败：定义了嵌套函数 `quality`，违反单函数约束。Design audit 却错误声称没有额外嵌套函数，说明模型自审不能替代 validator。
- `progress` 把位置、速度、姿态、角速度和接触全部合并为一个 state-quality difference，虽然覆盖任务需求，但 component statistics 无法判断是哪种行为职责失衡。
- 终局值为 ±100，运行时会裁剪为 ±20；截断还额外给予 `0.1*q_next`，可能奖励拖到时间上限。
- 合成转换：普通接近约 +0.393；接触变化约 +4.071；成功函数输出约 +100.020；失败约 -103.040；截断近目标状态约 +0.476。接触变化和终局事件明显压过普通进展。

结论：不能进入正式训练。

## B：Historical Expert Context

- 组件：`progress`、`fuel_penalty`、`terminal`。
- 通过静态接口验证；使用 bounded state scores、±2 terminal，并且截断不发终局奖，代码质量明显优于 A。
- 仍将位置、速度、角度、角速度和接触等权合并进 `progress`，不利于组件级故障定位。
- 合成转换：普通接近约 +0.027；同一步使用发动机后变为 -0.023；接触变化约 +0.319；成功约 +2.056；失败约 -1.992。
- 尺度存在关键错误：`phi` 位于约 [0,1]，因此整个 episode 的 potential difference 望远镜总和通常不超过约 1；而 `-0.05` 动作代价逐步累积。1000 步中若 30%–70% 使用发动机，累计为 -15 至 -35，远超 progress 与 terminal，容易诱导不行动。模型的 Design audit 将燃料累计估成约 -0.5，忽略了最大 episode 长度。

结论：B 是本轮较好版本，但在调整职责划分与燃料尺度前仍不适合作为正式初始奖励。

## A/B 结论

Historical Expert Context 本轮改善了数学有界性、代码合规和终局尺度，但没有解决独立可诊断性，并带来了严重的 episode 累积尺度失衡。它可以作为数学形式检查材料，不能作为固定任务知识直接注入的充分理由。

本轮同时证明了两项必要约束：第一，不能把所有成功相关量都压进一个综合 progress；第二，potential difference 必须按望远镜上界与逐步代价的 episode 累积比较。两项已补入 vNext prompts，未再次调用 API。
