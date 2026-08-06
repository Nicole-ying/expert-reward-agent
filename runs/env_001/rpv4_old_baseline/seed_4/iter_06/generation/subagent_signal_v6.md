# Subagent Research Signal

**训练过程**: Crash rate 100% all stages; generated_reward rose 0.146→1.115 but original_reward fixed -100/step; episode length stable ~70.

**组件健康**: contact_reward near-dead (0.7%); others 99-100% active; engine_penalty 47.2%; generated_reward mean -0.3214.

**奖励对齐**: Shaped reward excludes original crash penalty; agent exploits approach_reward (coefficient 20) while still crashing; no survival signal.

**异常检测**: Crash rate stagnation despite shaped reward growth; early exploitation of approach progress without landing learned.

**置信度**: `high`
