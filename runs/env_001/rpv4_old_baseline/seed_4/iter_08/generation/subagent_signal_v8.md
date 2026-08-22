# Subagent Research Signal

**训练过程**: Agent showed no survival learning. Crash rate remained 100% across all stages (70 steps avg, orig_reward -100/step). Shaped reward rose slightly (4.95→5.66) but behavior did not improve.

**组件健康**: all components had high nonzero rates except landing_success_reward (0.5% nonzero). approach_reward (mean 0.162/step) and survival_bonus (0.05/step) dominated. landing_success_reward is effectively dead.

**奖励对齐**: generated_reward (0.225/step) positive while original_env_reward (-1.683) negative after 100% crashes. Agent exploited approach_reward for positive total reward without task success. landing_success_reward rare, no guidance to landing.

**异常检测**: Early convergence to a policy that crashes on every episode yet earns positive shaped reward, with no later recovery or survival signal.

**置信度**: `medium`
