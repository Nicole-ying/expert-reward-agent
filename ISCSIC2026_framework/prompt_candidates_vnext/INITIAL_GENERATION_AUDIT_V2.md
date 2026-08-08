# Env-001 vNext 第二次初始生成测试审计

## 测试对象

- 运行目录：`runs/vnext_initial_ab/env001_prompt_quality_20260801_v2/`
- API 调用：1 次 Environment Card，1 次 card-only reward，1 次 historical-expert reward。
- 自动检查：环境卡 1--8 节完整；两份奖励均通过语法、名称绑定、单函数、接口和 component 数量检查。
- 本地合成检查：接近目标、远离目标、使用主发动机、成功终止、失败终止和静止六种转换均可执行。

## 环境卡结论

改进已经生效：最终卡片第 0 节只保留匿名任务描述，observation、action 和 termination 只在后续表格出现，不再复制整份 YAML。它也正确使用 `info["terminated"]`，没有再生成裸变量。

仍有一处过度推断：masked raw-step 返回 `truncated=False`，模型据此写成“no truncation happens”。实际训练通过 `gym.make` 创建环境，外层 `TimeLimit` 仍可能在最大 episode 步数处产生 `truncated=True`。因此只能说“环境内核不主动截断；完整运行仍可能由外层时间限制截断”。对应 prompt 和运行时接口说明已修正，未为此重复调用 API。

成功判定中的 `|position| < 0.1`、`|velocity| < 0.1` 是模型设计的启发式阈值，不是 masked source 提供的环境事实。它们必须通过训练反馈验证，不能在论文中写成已知官方成功条件。

## Card-only 奖励

四个实际 component：`goal_progress`、`fuel_efficiency`、`success_bonus`、`failure_penalty`。相对尺度不再都是 1：发动机代价为 -0.1，成功为 +10，失败为 -5；component 保存的是已加权贡献，求和关系清楚。

合成转换中，普通接近目标一步约为 +0.141；同一步使用发动机后约为 +0.041；成功转换约为 +10.071；失败转换约为 -4.847。代码合法，但 -0.1 的燃料代价已接近普通进展信号的量级，可能抑制任务必需的推进动作，需要 PPO 训练确认。

主要缺口是没有近目标速度、姿态或角速度引导。成功只在终止时通过猜测阈值识别，学习可能主要依赖距离差和稀疏终局事件。

## Historical-expert 奖励

四个实际 component：`goal_progress`、`success_bonus`、`failure_penalty`、`step_cost`。相对尺度为 clipped progress、+20、-10 和 -0.01。它修复了第一次样本中的 `original_reward`、嵌套函数和空 component 问题。

合成转换中，普通接近目标一步约为 +0.131；成功转换的函数输出约为 +20.061，但运行时会裁剪到 +20；失败约为 -9.857；静止为 -0.01。该版本用时间代价鼓励更快完成，但没有实现任务描述中的 thrust efficiency，也没有姿态/稳定性 shaping。

## Expert Context 的本轮证据

历史 Expert Context 将估算 prompt 长度从 5361 tokens 增加到 6801 tokens，约增加 27%。这次它带来了 progress clipping、较明确的终局尺度和 step cost，但同时丢失了燃料效率目标；两份设计都缺少连续的稳定着陆引导。因此本轮只能说明 Expert Context 改变了设计取舍，不能说明它必然提高奖励质量。

## 总结

第二次测试已经从“接口不合法”提升到“两份均可执行”，说明接口契约、完整性门控和 component 约束有效。但它们还只是可训练候选，不是高质量奖励的证明。下一项有意义的检验是以同一 PPO 预算训练两份代码，并比较 native score、成功率、终止分布和 component 贡献；在训练前不应仅凭 LLM 文本选胜者。
