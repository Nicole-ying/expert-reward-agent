# Subagent Research Signal

**训练过程**: Score improved slightly (8.7→9.4) but crash rate stayed 100% across all phases; episode length constant at 70. Agent never survived; only maximized shaped reward.

**组件健康**: Most components active >99%. landing_success_reward nonzero_rate=0.5% (zombie). engine_penalty 11.6%. approach_reward, speed_penalty, survival_bonus, etc. all near 100%.

**奖励对齐**: Mismatch: generated_reward mean +0.1296/step vs original_env_reward mean -1.6482/step. Agent exploits positive components without landing; shaped reward rose while env reward remained strongly negative.

**异常检测**: Persistent 100% crash rate despite increasing shaped reward. landing_success_reward almost never active (0.5%). Agent stuck in local optimum: optimizes approach/survival but fails landing.

**置信度**: `medium`
