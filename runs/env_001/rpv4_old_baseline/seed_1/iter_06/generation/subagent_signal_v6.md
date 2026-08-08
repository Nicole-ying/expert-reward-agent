# Subagent Research Signal

**训练过程**: Agent never learned to land or survive; avg_len=70 steps, crash_rate=100% from early to late training. avg_score flatlined at -7.3 across all stages. No reduction in original_env_reward (-100/step).

**组件健康**: contact_reward fired only 0.6% of steps (mean_when_active=1.329)—effectively dead. fuel_penalty active 19.6% of steps. All other components 100% nonzero, dominated by vel_penalty (-0.135/step) and shaping_reward (+0.031/step). generated_reward mean=-0.1055/step.

**奖励对齐**: generated_reward=-0.1055/step vs original_env_reward=-100/step—orders of magnitude gap. Shaped reward does not reflect task failure; original reward unchanged. Final eval mean -115.5 per episode confirms misalignment and no progress.

**异常检测**: Complete stagnation: same crash rate, episode length, and score across all training chunks. No learning occurred. contact_reward virtually absent, landing signal missing.

**置信度**: `high`
