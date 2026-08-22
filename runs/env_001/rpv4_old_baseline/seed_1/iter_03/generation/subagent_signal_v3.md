# Subagent Research Signal

**训练过程**: Agent never reduced crash rate (100% throughout). Average episode length slightly oscillated (73→71→71). Generated reward per step improved from -1.628 to -0.687, but original env reward stayed constant at -100/step. Final evaluation mean episode length 68.35, all terminated.

**组件健康**: Landing_bonus and unbalanced_penalty nearly dead (nonzero rate 1.6%). Thrust_cost active 47.4% of steps. Other components (approach_reward, vel_penalty, stability_penalty, original_env_reward) fired at 100% nonzero rate. Dominant penalty components throughout training.

**奖励对齐**: Generated reward per step improved while original env reward remained -100/step, indicating the shaped reward guides the agent to reduce movement penalties but not to achieve landing. No exploitation of reward shaping observed, but no task success either.

**异常检测**: Early convergence to a crashing-only policy; crash rate never decreased despite shaped reward improvements. No sudden divergence or value explosion.

**置信度**: `medium`
