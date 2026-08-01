# Env-001 vNext 初始生成审计

## 审计对象

本次本地串联使用同一张 Environment Card 生成两份初始奖励：

- `card_only`：仅使用精简环境卡。
- `historical_expert`：额外注入 paper-v4 历史 Expert Schema Context。

本地输出目录：`runs/vnext_initial_ab/env001_prompt_quality_20260801_rerun/`。

## 环境卡

有效部分：

- 正确保留匿名任务描述，并将任务识别为 `goal_reaching`。
- Observation 和 Action 采用表格，索引关系清楚。
- 将 crash、out-of-bounds、settled 和 truncation 分开讨论。
- 选择 `goal_delta` 作为主进展方向，符合目标到达任务的 success semantics。

发现的问题：

1. Environment Analyzer 只看到了 step source，没有看到 wrapper 的真实 reward interface，因此错误地把裸变量 `terminated/truncated` 视为 reward 可访问。真实接口是 `info["terminated"]`、`info["truncated"]` 和 `info["done"]`。
2. `info["terminated"]` 只表明发生终止，不提供 crash/out-of-bounds/settled 的具体原因；环境卡对部分 failure 的“可识别性”表述过强。
3. 成功阈值（位置、速度、角度）并非 task spec 明示事实，只能作为设计启发，不能写成已知环境契约。
4. 模型输出在第 6 节中途截断，缺少第 7–8 节。旧脚本仍继续生成奖励，这是不安全的。

已修复：Environment Analyzer 的 User Prompt 现在包含真实 `REWARD_INTERFACE_CONTRACT`；脚本要求第 1–8 节完整，否则停止且不调用 Reward Generator。

## Card-only 初始奖励

优点：

- 四个 component 职责清楚：progress、stability、energy、terminal outcome。
- `goal_progress` 使用 distance delta，方向与任务目标一致。
- 没有使用 original reward 或虚构 `info["success"]`。

阻断问题：

- 代码使用未定义裸变量 `terminated`，运行时会触发 wrapper fallback；原 validator 只做 AST compile，没有发现未绑定名称。
- 注释声称使用 tanh/bounded stability，实际代码只是未界定的绝对值惩罚。
- 设计说明估计 `goal_progress` 为 0.01–0.1，而 engine penalty 固定为 -0.1；惩罚可能与主进展同量级甚至更强，与“正向信号主导”的叙事冲突。
- ±100 terminal bonus 会被当前 wrapper 的 reward clip 截到 ±20；Prompt 没有感知这一运行时事实。
- success thresholds 是启发式猜测，不是接口事实。

结论：概念结构可读，但当前代码不可训练。

## Historical-expert 初始奖励

优点：

- 模型主动发现了函数签名与 termination signal 的冲突。
- progress、stability 和 state-based success 的职责较明确。

阻断问题：

- 写出 `terminated = original_reward`，直接违反禁止使用 original reward 的规则。
- 定义嵌套 helper function `distance`，违反单函数代码契约。
- 保留恒为 0 的 `failure_penalty` 只为凑齐结构，违反“component 必须实际承担职责”的原则。
- 大量自我争论被写进代码注释，降低可执行代码的简洁性和审计性。
- 没有实现任务描述中的 thrust-efficiency 职责，但保留了无效 failure component。

结论：历史 Expert Context 没有改善本次代码合规性，反而使 Prompt 估算长度从 4265 tokens 增加到 5705 tokens（约 +34%），并产生更冗长、冲突更明显的输出。

## 关于 Expert Context 的结论边界

本次结果不能证明“专家知识永远无用”，也不能比较最终 PPO 性能，因为两份代码都未通过接口审计。它能够支持的较窄结论是：

- paper-v4 的整包 Expert Schema 与精简 vNext 环境卡存在字段和职责上的重复/冲突；
- 在本次样本中，整包注入没有带来更好的初始代码质量；
- 下一轮应先生成两份合法奖励，再在相同 PPO 预算下比较 `card_only`、精简定向 expert context 和历史整包 context。

## 已实施的脚本修复

- 最终环境卡第 0 节只保留 `task_description` 原文，不再复制与后续表格重复的整份 YAML。
- Environment User Prompt 注入真实 reward interface contract。
- 明确结束标志必须通过 `info` 访问，禁止裸变量。
- 将 runtime reward clip 写入接口契约，要求事件 bonus 按实际 clip 设计。
- 统一 component 表达：component 保存已经加权的真实贡献，权重只应用一次，`total_reward` 直接求和。
- 要求逐项估计正常步/事件步尺度，禁止注释声称 bounded/tanh 而代码没有实现。
- 环境卡缺少任一第 1–8 节时停止生成。
- validator 检测未定义名称。
- validator 拒绝嵌套/额外函数。
- 支持 `--resume-run`，只补未完成阶段，避免重复 API 花费。
