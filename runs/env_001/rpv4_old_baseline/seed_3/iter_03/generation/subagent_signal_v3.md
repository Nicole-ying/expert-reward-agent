# Subagent Research Signal

**训练过程**: Episode length grew from 290→466, score from 556→1064, crash rate fell 65%→40%. Original env reward per step improved from -78.7→-59.8. Final eval episodes all length 1000, no early termination. Steady progress but task not fully solved.

**组件健康**: soft_landing dominates (per-step mean 2.19, nonzero rate 51.5%, when active mean 4.25). velocity_damping (-0.08) and orientation (-0.05) always active, small negative. progress 0.0006 (99.9% nonzero, negligible). generated_reward total per step 2.07, with soft_landing as main driver.

**奖励对齐**: Strong gap: generated_reward per step 2.07 vs original_env_reward -0.04. Soft landing provides large positive reward not reflected in original env reward. In trend windows, generated_reward was negative despite overall positive mean, suggesting soft_landing may fire in training but not reliably in sampled eval episodes, leading to potential exploitation and misalignment.

**异常检测**: Trend windows show negative generated_reward (-3.2→-0.2) while overall training stats show positive mean 2.07, indicating sampling mismatch. Soft_landing high weight (9.0) may encourage repeated triggering without episode termination; all eval episodes truncated, no termination, suggesting cycle behavior exploiting shaped reward.

**置信度**: `medium`
