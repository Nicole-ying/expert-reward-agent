# V3 A 初始奖励 1M PPO 诊断

## 实验配置

- reward：`ISCSIC2026_framework/runs/vnext_initial_ab/env001_paperv4_pruned_20260801_v3/rewards/card_only/reward_v1.py`
- fresh PPO，seed 0，4 个并行环境
- 训练预算：1,000,000 steps（SB3 rollout 对齐后实际记录 1,003,520 steps）
- native evaluation：固定 seeds 10000–10019，共 20 episodes
- 训练耗时：1,140.6 s（约 19.0 min，CPU）
- reward clip：±20

## Native evaluation

| 指标 | 结果 |
|---|---:|
| mean native score | -121.544 |
| median | -121.709 |
| sample std | 10.513 |
| min / max | -147.383 / -105.977 |
| 正分回合 | 0/20 |
| 达到 200 分回合 | 0/20 |
| terminated / truncated | 20/20 / 0/20 |
| mean episode length | 68.4 |
| early failure（<150 steps 且 score < -50） | 20/20 |

## 最终策略的奖励组成

| component | episode sum mean | magnitude share | active rate |
|---|---:|---:|---:|
| progress | +1.1186 | 16.7% | 100.0% |
| stable_landing | -2.7500 | 82.8% | 1.5%（逐步口径） |
| thrust_cost | -0.0360 | 0.5% | 5.3% |

## 机制诊断

该策略没有表现出“燃料成本累计过强”所预期的长 episode 或高负燃料贡献。相反，策略平均只承担 -0.036 的燃料成本，说明评估时发动机动作极少；它仍能从 `progress` 获得约 +1.119，因为自然下落本身会缩短到目标的距离。

因此，当前奖励允许以下代理策略：

1. 少用或不用发动机，降低 `thrust_cost`；
2. 利用自然动力学快速接近目标，获得 `progress`；
3. 缺少飞行阶段的速度、姿态和角速度质量约束，最终在约 68 步终止；
4. 失败惩罚不足以让策略学习为安全接触主动控制。

`stable_landing` 在 20 回合中的均值为 -2.75，而该组件只能取 +10 或 -5。这对应约 3 次 +10 和 17 次 -5。然而全部 native score 都低于 -105，说明启发式成功判据把部分 native 失败误判成了安全着陆。该组件不仅稀疏，还存在成功假阳性。

## 对下一次有界修复的证据

本结果定位到的首要问题不是简单调大/调小某个全局系数，而是 `stable_landing` 职责的可观测语义不完整：

- 仅在 terminal state 检查位置、速度、姿态和双接触，仍不能可靠对应 native 成功；
- `progress` 只衡量位置接近，无法区分受控下降与被动坠落；
- 被延后的速度/姿态质量至少需要以一个高层、门控的 `approach_and_landing_quality` 职责进入修复候选，而不是拆成多个低层惩罚组件；
- 终局正奖励在可靠成功证据不足时不应继续作为确定性 +10 发放。

这正是 CREATE 所需要的训练证据：component-level statistics 将失败定位到“位置进展可被被动坠落利用 + 终局成功判据假阳性”，下一轮应做一次有界的职责/骨架修复，而不是盲目增加组件或全局调权。
