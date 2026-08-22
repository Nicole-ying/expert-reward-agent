# Subagent Research Signal

**训练过程**: No improvement in survival: crash rate 100% throughout, avg_len constant at 71. Score slightly improved from -26.9 (early) to -22.3 (mid) then fell to -24.3 (late). Generated reward per step rose from 9.678 to 10.589, but original env reward stayed at -100.0/step (trends) or -1.5466/step (components, inconsistent). Agent never learned to land.

**组件健康**: Active components: speed_penalty (100% nonzero), angvel_penalty (99.5%), survival_penalty (100%), approach_reward (100%), angle_penalty (100%), engine_penalty (56%). Zombie components: base_contact_bonus (1.7% nonzero), soft_contact_bonus (0.6% nonzero) — landing bonuses never triggered. Generated reward mean -0.3456/step dominated by speed_penalty (-0.5226) and survival_penalty (-0.1).

**奖励对齐**: Poor alignment: generated_reward (-0.3456/step) far less negative than original_env_reward (-1.5466/step) per component stats. Shaped reward overestimates agent's performance, likely causing reward exploitation. Agent achieves less negative generated reward without landing, so survival/landing behaviors not incentivized. 100% crash rate in all phases confirms exploitation.

**异常检测**: Monitor log reports original_env_reward = -100.0/step in trends vs -1.5466/step in component stats — inconsistency suggests monitoring error or different units. Crash rate may be reported as >100% (likely 100%). Constant episode length 71 despite crashes.

**置信度**: `medium`
