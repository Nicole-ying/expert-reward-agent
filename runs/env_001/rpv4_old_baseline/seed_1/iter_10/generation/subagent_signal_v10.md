# Subagent Research Signal

**训练过程**: Agent never learned to survive: crash_rate=100% across all phases. avg_score improved slightly from -4.7 to -3.6, but episode length flat at ~70. generated_reward/step increased from 7.503 to 7.809 while original_env_reward/step remained -100, indicating minimal task progress.

**组件健康**: Active components: shaping (100% nonzero), survival_bonus (100%), ground_danger_penalty (16.6%), fuel_penalty (33.2%), angle_penalty (2.7%). Dead/zombie components: crash_penalty (0.0%), success_bonus (0.5%), contact_continuous (0.7%), angvel_penalty (0.8%).

**奖励对齐**: Severe misalignment: generated_reward per step (-0.0590) heavily offset from original_env_reward (-1.6462). Shaped reward improved while original reward flat, suggesting exploitation: agent maximizes generated_reward via shaping without learning safe landing (crash_rate 100%). No evidence of original reward improvement.

**异常检测**: Permanent crash plateau: 100% crash rate persisted from early to late training. Reward exploitation: generated_reward diverged positively from original. Multiple reward components effectively dead during training.

**置信度**: `high`
