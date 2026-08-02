# paper-v4-pruned Env-001 A/B 审计（V3）

## 测试协议

- 运行目录：`runs/vnext_initial_ab/env001_paperv4_pruned_20260801_v3/`
- A：Environment Card only。
- B：同一张 Environment Card + historical Expert Context。
- DeepSeek API 正常调用 3 次：环境卡、A、B各一次；无重试。
- 两组都通过静态代码 validator。本轮未运行 PPO。

## Environment Card

更新后的职责优先级生效。卡片选择三个初始职责：

1. `progress_to_target`：终局前的稠密方向信号；
2. `thrust_cost`：独立的控制成本；
3. `stable_landing`：终局安全着陆质量。

`continuous_orientation_penalty` 和 `early_time_penalty` 被延后。相比 V2，这次没有为了满足组件预算而删除主要进展信号，也没有把逐步时间惩罚当作“快速完成”的替代品。

## A：card-only

生成三个组件：`progress`、`thrust_cost`、`stable_landing`。

| 合成转移 | 总回报 |
|---|---:|
| 停滞 | 0.000000 |
| 大幅接近、不用发动机 | 0.565685 |
| 大幅接近、使用发动机 | 0.555685 |
| 小幅接近、使用发动机 | 0.004142 |
| 远离目标 | -0.565685 |
| 启发式成功终局 | 10.070711 |
| 明显失败终局 | -5.067554 |
| 仅 truncated | 0.000000 |

三个关键反事实成立：进展优于停滞；合理控制后的进展仍优于停滞；远离目标为负。组件职责清晰，可以分别调整距离势函数、燃料尺度和终局判断，适合作为 CREATE 后续单组件诊断与修复的起点。

剩余风险：

- fuel 最坏 1000 步累计为 -10，而距离差的 episode 总量受初末距离限制；实际是否压制控制仍需看动作激活率和训练结果。
- 代码只检查 `terminated`，在两个结束标志同时为真时仍会发终局事件。
- 成功阈值是启发式的，可能过严；这属于可由终局激活率和 native outcome 定位的修复点。

## B：card + historical Expert Context

生成三个组件：`progress_to_target`、`thrust_cost`、`stable_landing`。

| 合成转移 | 总回报 |
|---|---:|
| 停滞 | 0.000000 |
| 大幅接近、不用发动机 | 0.009600 |
| 大幅接近、使用发动机 | -0.000400 |
| 小幅接近、使用发动机 | -0.009682 |
| 远离目标 | -0.009600 |
| 启发式成功终局 | 3.788029 |
| 明显失败终局 | -0.001200 |
| 仅 truncated | 0.000000 |

B 的 `w_progress=0.01` 与每次发动机动作 `-0.01` 尺度不匹配。即使测试转移把平方距离从 1.28 降到 0.32，使用发动机后的净回报仍为负。轻微但必要的控制更明显为负，违反 prompt 的 `progress_with_control_cost > idle` 反事实要求。

此外，B 的终局项只给予正 bonus，没有明确失败 penalty；明显撞毁几乎只得到 -0.0012。它还把距离差错误描述为“不是 potential-based shaping”，说明文字审计与代码数学语义不一致。

## A/B 结论

1. 更新后的 Environment Card 已解决“五组件太多”和“删除稠密进展信号”两个问题。
2. A 的结构与局部尺度明显优于 B，已经接近可用于小预算 PPO smoke test 的初始奖励。
3. Historical Expert Context 没有改善结构，反而压低进展尺度并削弱失败信号。本轮支持从正式初始奖励链路移除它，而不是把它作为必要输入。
4. 两组都没有严格实现 `terminated and not truncated`。候选 Reward Generator prompt 已把该条件升级为代码级硬约束，并把三类反事实从“审计建议”升级为输出接受条件。

因此，若进入训练验证，应优先使用 A，而不应训练 B。A 的训练目的不是证明它已经完美，而是验证 CREATE 能否通过训练证据识别燃料尺度或终局阈值问题，并进行有界单组件修复。
