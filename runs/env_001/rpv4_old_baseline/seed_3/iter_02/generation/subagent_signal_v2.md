# Subagent Research Signal

**训练过程**: Episode length rose from 385 (early) to 599 (late), crash rate fell from 62% to 38%. Average score improved from 30.4 to 49.5. Per-step generated_reward shifted from -0.028 to 0.027. Agent learned to survive longer but original_env_reward remained deeply negative (-67.5 to -44.4).

**组件健康**: All components active: total_reward/generated_reward nonzero 100%. contact_bonus (nonzero 69.7%, mean 0.0674) dominates reward. progress scaled (10x) mean 0.0254, small. Penalties (speed, angle, angvel) active but minor. No dead components.

**奖励对齐**: training per-step generated_reward vs original_env_reward gap large: orig always negative (<-44), gen near zero or positive. In evaluation mean cumulative generated_reward=151.3, but no eval orig_reward to compare. contact_bonus dominates, possibly exploited by agent to survive without truly solving task.

**异常检测**: Not reported.

**置信度**: `medium`
