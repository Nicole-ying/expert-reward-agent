# seed 1 未搜索到合格奖励函数的根因审计

## 结论

seed 1 并非单纯因为 PPO 随机性或搜索预算不足而失败。主要问题是终局语义从“不可精确区分”越界成了未经校准的 success/failure 启发式，且旧反馈只有跨 episode 聚合值，Reflection Agent 无法识别假阳性。两次 fresh restart 的合规重试还发生了训练证据错位。因此，本轮 `r1` 应视为调试实验，不能作为最终五种子主实验。

## 证据链

1. 环境接口只暴露合并后的 `info["terminated"]`，不暴露具体结束原因或精确 success/failure 标签。
2. seed 1 的旧环境卡虽然承认阈值未知，却仍允许把近目标、低速、姿态和接触阈值组合直接当作 success，并在后续 fresh restart 的环境卡中把 TimeLimit 超时写成 failure。三张环境卡的语义并不一致。
3. iter 1 的 `touchdown_success` 极少触发，策略 20/20 提前结束，native mean 为 -123.35。
4. iter 2 把稀疏终局信号改成每步 `touchdown_proxy` 后，策略学会持续占据代理状态：20/20 全部 truncated，平均长度 1000，native mean 仍只有 -15.06。
5. iter 7 的 `terminal_event` 在 20 个 episode 上平均为 +6。由于单次取值为 ±20，这意味着启发式给出了大量正终局判断；但同批 native mean 仍为 -46.41。旧反馈没有逐 episode 对齐表，Reflection 仍把正 terminal share 当成“更多成功”的方向性证据。
6. iter 8 又将主信号改为每步 proximity，17/20 episode 达到时间上限，平均长度 937.55，native mean -22.14，重新出现 horizon/proxy farming。
7. fresh restart 标志在验证前被清零，导致 fresh draft 验证失败后误走 Reflection 修订路径，并读取上一轮甚至上上轮训练反馈；iter 6 与 iter 9 的重试证据因此不对应当前草稿。

## 已实施修复

- Environment Card 必须输出 `exact / derived_reliable / heuristic_only / unavailable` 四级终局可靠性，以及 permitted reward use、假阳性/假阴性和逐 episode 校准方案。
- Reward Generator 与 Reflection Agent 将该边界视为硬约束，不再把 `heuristic_only` 写成真实 success/failure。
- external/native evaluation 保存每个 episode 的最终 observation 与每个奖励组件累计值；Reflection 输入新增逐 episode 对齐表和终局假阳性、proxy accumulation 警告。
- fresh generation 的来源标志在本轮开始时冻结；代码验证重试始终由原生成器修复，不再使用错位的 Reflection 证据。
- Environment Card 在同一实验以及五个 seed 之间只生成一次并冻结；fresh restart 只重建奖励，不重新采样环境语义。

## 干净重跑原则

1. 保留旧 `r1` 作为审计与调试记录，不覆盖。
2. 用新实验前缀从 seed 0 开始运行五个 seed；首个 seed 生成并验证共享 Environment Card，seed 1–4 复用该卡。
3. 每个 seed 最多 10 次完整 1M-step PPO 训练，每轮使用固定 20-episode native evaluation。
4. 首次 native mean >= 200 后立即停止该 seed；不执行 post-threshold probe。
5. 最终报告 trainings-to-threshold、每 seed 最佳 native mean、逐 episode 终局对齐和每次实际组件改动。
