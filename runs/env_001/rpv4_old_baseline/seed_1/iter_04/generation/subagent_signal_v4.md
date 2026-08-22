# Subagent Research Signal

**训练过程**: Agent improved over training: avg_len from 204 to 303, avg_score from 346 to 1192, crash_rate from 69% to 49%. Later episodes are longer with higher shaped reward. Original env reward per step remained very negative (-92.5 to -84.4), showing minimal improvement.

**组件健康**: All components active. landing_bonus sparse (nonzero_rate 18.4%) but high when active (mean 327.6). altitude_reward and vel_penalty always active. approach_reward very small (0.037). thrust_cost active 78.3% of steps.

**奖励对齐**: Severe misalignment. Training avg generated_reward +2.68/step, but original_env_reward -0.40/step. Evaluation mean reward -185.7 despite all episodes terminated. Agent may be exploiting shaped reward (e.g., altitude bonus near y=0) without achieving soft landing.

**异常检测**: Original env reward per step remains consistently negative and does not improve with shaped reward. generated_reward turns positive late while original reward stays below -80. Crash rate drops, but eval episodes still score very negative.

**置信度**: `medium`
