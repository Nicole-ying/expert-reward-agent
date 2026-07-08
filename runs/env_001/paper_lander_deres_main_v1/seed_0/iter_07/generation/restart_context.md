# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- approach_quality_reward + attitude_penalty + contact_bonus + progress_reward
- approach_quality_reward + attitude_penalty + landing_proxy + progress_reward
- approach_quality_reward + attitude_penalty + progress_reward
- attitude_penalty + landing_quality_reward + progress_reward

上述骨架在前序迭代中已尝试但未取得突破。
请基于训练证据选择改进方向：
- 如果认为同一骨架仍有可修复空间（如系数调节、条件化约束），可以继续在当前骨架上修改。
- 如果诊断表明当前骨架存在结构性问题（如信号冲突、梯度消失），请从 expert_reward_context.md 中选择不同的数学形态。
- 不要机械重复已失败的骨架。
