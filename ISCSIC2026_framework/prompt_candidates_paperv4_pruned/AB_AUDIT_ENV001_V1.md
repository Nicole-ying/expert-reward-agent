# paper-v4-pruned Env-001 A/B 审计

## 协议

- 运行目录：`runs/vnext_initial_ab/env001_paperv4_pruned_20260801_v1/`
- 共享 1 张 Environment Card；A 为 card-only；B 为 card + Historical Expert Context。
- 初始三次调用中 A 的服务响应在函数中途截断；随后只对已确认 invalid 的 A 做一次定向重试。因此本轮实际共 4 次 DeepSeek 调用，没有重跑环境卡或 B。
- Prompt tokens：A 约 4932，B 约 6372；Historical Expert Context 增加约 29.2%。

## Environment Card

卡片恢复了 paper-v4 的动态 role decomposition，同时没有恢复专家画像或固定任务分类。它得到五个由当前任务证据推导的职责：proximity、velocity damping、orientation stabilisation、fuel efficiency 和 conditional safe landing。每个 role 都具有独立 legal signals、temporal semantics 和 failure evidence。Runtime contract 正确保留 ±20 clip 与 1000 步上限。

存在两点可继续压缩：部分动作方向被写成 `likely` 推测，实际 reward 不需要这些方向；卡片较长，但主要长度来自可用于反思的 role/failure evidence，而不是专家画像。

## A：card-only

- 通过静态验证；五个 components：`proximity`、`velocity`、`orientation`、`fuel`、`safe_landing`。
- 合成转换：普通接近并稳定约 +0.814；同一步使用发动机约 +0.794；成功约 +15.480；失败撞击约 -1.400；截断近目标状态约 +0.924（没有 terminal event，只保留合法 transition improvements）。
- 优点：职责完全分开，燃料单步代价没有压过有益进展，成功事件在 ±20 clip 内。
- 风险：权重相对激进；五个组件超出推荐 2–4，但每个都有独立 role 依据。最大 episode 中 fuel 最坏累计为 -20，仍需通过训练激活率判断是否过强。

## B：Historical Expert Context

- 通过静态验证；五个 components：`proximity`、`velocity_damping`、`attitude_stabilisation`、`fuel_efficiency`、`safe_landing_bonus`。
- 合成转换：普通接近并稳定约 +0.178；同一步使用发动机约 +0.158；成功约 +8.107；失败撞击约 -0.531。
- 与 A 相比采用更保守的 shaping 与 terminal 权重；职责结构几乎相同，说明 role decomposition 来自剪枝版 Environment Card，而不是必须依赖 Expert Context。
- fuel 同为 -0.02，1000 步最坏累计为 -20；实际是否导致不行动必须看 action active rate 和 native outcome。

## 共同问题

- 两份 terminal block 都检查 `terminated`，但没有显式加入 `not info.get("truncated", False)`。若运行时同时出现 terminated 与 truncated，可能错误发放 safe-landing bonus。正式训练前应进行确定性语义修复。
- safe-landing 是稀疏 conditional component，首轮可能完全不激活；这不是空组件，因为它有合法触发路径，但 Reflection Agent 必须把 zero active rate 解释为阈值过严、策略未到达或终局语义不可靠，而不是自动提高权重。
- 两份 Compact design audit 都在表格中途被服务截断，代码完整但文字审计不完整。后续 runner 应把 audit completeness 与 code validity 分开记录。

## 结论

本轮支持 paper-v4-pruned 路线：它恢复了适合 CREATE 单组件反思的独立职责骨架，同时保留了简化探索得到的 runtime 与尺度约束。Historical Expert Context 影响了公式和权重的保守程度，但不再决定组件是否可诊断。A/B 均比此前 all-in-one progress 更适合作为迭代起点；在修复 terminal/truncated 条件并增加确定性 episode-bound lint 后，可进入小预算训练验证。
