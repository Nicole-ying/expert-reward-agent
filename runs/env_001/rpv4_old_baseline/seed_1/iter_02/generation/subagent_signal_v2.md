# Subagent Research Signal

**训练过程**: No improvement: crash rate held at 100% across early/mid/late stages (all 4646–4647 episodes). Mean episode length constant at 72 steps; generated reward per step varied (early 0.642, mid 1.158, late 0.724) but original env reward remained −100/step. Agent never learned to survive.

**组件健康**: approach_reward dominated (100% active, mean 1.56). stability_penalty (100%, −0.15) and thrust_cost (45.9%, −0.03) active but small. landing_bonus nearly dead (1.7% active; when active, mean 46.5) – effectively a zombie component.

**奖励对齐**: Severe gap: per-step generated_reward mean +1.559 versus original_env_reward −1.345. Evaluation mean original reward −84.0, all 20 episodes terminated early (avg length 69.7). Positive shaping rewards did not translate to task success – clear exploitation.

**异常检测**: Agent failed to reduce crash rate from start to finish despite varying generated reward, indicative of reward hacking or a frozen policy.

**置信度**: `high`
