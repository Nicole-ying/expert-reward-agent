# paper-v4-pruned Env-001 A/B 审计（V2）

## 测试协议

- 运行目录：`runs/vnext_initial_ab/env001_paperv4_pruned_20260801_v2/`
- A：Environment Card only。
- B：同一张 Environment Card + historical Expert Context。
- DeepSeek API 共正常调用 3 次：环境卡 1 次、A 1 次、B 1 次；没有截断重试。
- 本轮只验证初始奖励生成、静态代码约束和合成转移，不包含 PPO 训练，因此不能据此主张真实性能。

## 数量约束结果

2–4 个组件预算已生效。环境卡选中四个职责：

1. `safe_landing_and_settle`
2. `soft_landing_velocity`
3. `time_efficiency`
4. `fuel_efficiency`

A 与 B 均生成四组件、单函数代码，并通过静态 validator。Expert Context 没有改变职责结构，只改变了系数与局部公式。

## A：card-only

组件为 `landing_success`、`touchdown_velocity_penalty`、`time_penalty`、`fuel_penalty`。

| 合成转移 | 总回报 |
|---|---:|
| 远处无进展、无动作 | -0.005 |
| 明显接近目标且稳定 | -0.005 |
| 明显接近目标且使用发动机 | -0.015 |
| 启发式成功终局 | 14.995 |
| 明显失败终局 | -10.005 |
| 仅 truncated | -0.005 |

最大 1000 步时，time 最坏累计为 -5；若每步使用发动机，fuel 最坏累计为 -10。两者联合可达 -15，与成功事件 +15 同量级。

## B：card + historical Expert Context

组件为 `safe_landing`、`soft_landing_velocity`、`time_efficiency`、`fuel_efficiency`。

| 合成转移 | 总回报 |
|---|---:|
| 远处无进展、无动作 | -0.020 |
| 明显接近目标且稳定 | -0.020 |
| 明显接近目标且使用发动机 | -0.070 |
| 启发式成功终局 | 17.970 |
| 明显失败终局 | -15.020 |
| 仅 truncated | -0.020 |

最大 1000 步时，time 最坏累计为 -20；若每步使用发动机，fuel 最坏累计为 -50。联合最坏累计 -70，显著超过成功事件 +18。`reward_clip=20` 是逐步总回报裁剪，不能限制 episode 累积，因此不能消除这一风险。

## 共同缺陷

1. **缺少前终局的任务进展信号。** 明显朝目标靠近与远处停滞得到相同回报；使用必要控制反而更低。对长时域任务，这会把初始学习压力主要放在稀疏终局探索上。
2. **四组件是“数量正确、职责选择错误”。** 接触速度只在接触后激活，终局项只在 terminated 时激活；时间与燃料则持续给负值。组件预算不应通过删除主要进展职责来满足。
3. **效率项的 episode 尺度未真正约束。** B 的逐步成本尤其激进；输出中的 scale audit 没有阻止代码违反审计意图。
4. **终局/截断边界仍不严谨。** 两组都只检查 `terminated`，当 `terminated=True` 与 `truncated=True` 同时出现时仍会发放成功/失败事件。B 还使用 `info['terminated']`，若字段缺失会抛出 `KeyError`，不如契约式 `info.get(..., False)` 稳健。

## 结论

本轮证明了组件预算规则有效，但也否定了当前角色优先级：不能把“2–4 个组件”理解为从五个职责中任选四个。初始 reward 至少应保留一个在普通、非终局转移上可观测的主要任务进展职责；终局、接触质量和逐步成本不能替代它。

因此，当前 A、B 都不适合直接进入正式 PPO 对比。候选 prompt 已根据本轮证据加入：前终局进展优先级、成本项延后规则、三类最小反事实检查，以及时间/燃料联合 episode-bound 审计。下一轮 A/B 应先确认这些确定性约束生效，再决定是否花训练预算。
