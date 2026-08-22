# Subagent Research Signal

**训练过程**: Agent never learned to survive: crash rate 100% across early, mid, late phases. Avg score improved slightly (-49.3 → -45.6) due to shaped reward gains, but task failure persisted. Episode length stable (~70).

**组件健康**: Survival_penalty, progress_reward, orientation_penalty all active (nonzero 100%). Engine_penalty 42% nonzero. Soft_contact_reward nearly dead (1.6% nonzero), mean 37.18 when triggered — likely a zombie component.

**奖励对齐**: Generated reward (-0.67/step) is less punitive than original env reward (-1.65/step). Agent improved shaped reward (via lower orientation/progress penalties) without reducing crash rate, indicating reward exploitation. Original crash penalty not reflected in shaped reward.

**异常检测**: 100% crash rate persisted throughout training, with no downward trend. Agent plateaued in task success while shaped reward kept rising — strong exploitation signal.

**置信度**: `medium`
