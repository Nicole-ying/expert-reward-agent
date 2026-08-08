# Subagent Research Signal

**训练过程**: Agent never learned to survive: crash_rate=100% throughout (early/mid/late 4640 eps). Avg episode length slightly decreased 73→71, avg score marginally improved -189.9→-185.5. Generated reward/step rose 1.671→3.070 while original_env_reward/step stayed -100, indicating no true task progress.

**组件健康**: Penalties all active (nonzero >99%): speed_penalty (-2.07/step) and original_env_reward (-1.57) dominate. engine_penalty 56.5% active. Contact bonuses nearly dead: base_contact_bonus and soft_contact_bonus each 1.6% nonzero, mean when active 10.0 and 17.9 respectively.

**奖励对齐**: Shaped total_reward mean -2.60/step vs. original_env_reward -1.57/step, so shaped is more negative. Generated reward/step improves while original reward unchanged, exposing reward gap. Landing bonus never triggers; agent exploits by minimizing penalties without learning to land.

**异常检测**: Persistent 100% crash rate after 13,920 episodes; no survival signal emerged. Contact bonuses delivered only 1.6% of steps, rendering landing incentives ineffective. Possible early convergence to local optimum (quick crash) rather than genuine skill.

**置信度**: `medium`
