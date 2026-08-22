# Env-001 vNext 第四次 A/B 初始生成审计

## 协议与运行时契约

- 运行目录：`runs/vnext_initial_ab/env001_prompt_quality_20260801_v4/`
- 调用：1 次 Environment Card；A 为 card-only；B 为 card + paper-v4 Historical Expert Context。
- Prompt tokens：A 约 5458，B 约 6898；Expert Context 增加约 26.4%。
- 两个模型均直接收到 `reward_clip=[-20,+20]` 和 `maximum episode steps=1000`。

## 已验证生效的修复

- Environment Card 正确保留 ±20 clip 和 1000 步上限。
- 两份奖励都把启发式终局阈值标为设计假设，而不是官方事实。
- terminal 使用 ±10，没有再写被 clip 浪费的 ±100。
- truncated episode 不再获得额外正终局奖励。
- 两份奖励都显式估计或响应了 fuel、terminal 和 progress 的尺度关系。

## A：card-only

- 组件：`progress`、`fuel_penalty`、`terminal_bonus`。
- 自动验证失败：再次生成嵌套函数 `potential`。Prompt 已明确禁止，说明该约束必须依靠 validator 和自动修复重试，不能继续靠增加文字解决。
- 合成转换：普通接近约 +0.206；同一步使用发动机约 +0.186；接触改善约 +0.196；成功约 +10.121；失败约 -10.951；截断近目标约 +0.252（仅来自合法 progress，没有终局奖励）。
- `-0.02` fuel 在普通有益步骤中没有压过 progress，单步关系比 v3 明显改善；最坏 1000 步累计为 -20，仍需训练统计确认。
- `progress` 仍把位置、速度、姿态和角速度合为一个职责，不利于定位独立行为缺陷。

结论：语义和尺度比 v3 A 好，但代码不合法，不能训练。

## B：Historical Expert Context

- 组件：`progress`、`fuel_penalty`、`terminal_outcome`。
- 通过静态代码验证，没有嵌套函数；terminal 为 ±10 且截断为 0。
- 输出在 Design audit 开头被截断，缺少完整的尺度、望远镜上界和 component budget 自审；当前 validator 只验证代码，因此仍显示 valid。
- 合成转换：普通接近约 +0.0256；同一步使用发动机后约 -0.0344；接触改善约 +0.0462；成功约 +10.018；失败约 -10.213。
- `fuel_penalty=-0.06` 高于普通有益步骤的 progress，且 1000 步最坏累计为 -60；即使只有 30% 动作激活也约为 -18。该尺度容易诱导不行动，说明 runtime contract 虽成功传入，模型仍未正确执行 episode 累积审计。
- `progress` 同样合并多个独立行为职责，组件级诊断粒度不足。

结论：代码合法，但燃料尺度和诊断粒度仍不足，不建议直接训练。

## 总结

第四轮证明 runtime contract 的直接注入有效，也修复了 ±100 和截断正奖励问题；但 Prompt 文本不能可靠保证代码形态和数值推理。下一步不应继续无限增加文字，而应采用三层机制：Prompt 负责设计原则，validator 硬拒绝非法代码，自动修复重试只处理明确错误；尺度方面在调用后增加确定性的本地 episode-bound 审计。Historical Expert Context 连续提高代码合规率，但没有稳定提高奖励尺度质量。
