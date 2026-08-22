# Subagent Research Signal

**Key Findings**: Eval score -87.2, ep len 143.7, all terminated (crashes). Early terminal 9/20 with score<-50. Total reward per-step small positive (0.015), but cumulative env reward negative.

**Component Anomalies**: landing_contact_reward active_rate 11.9% but signed_share 26.5%, sparse. action_cost active 45.9%, mean -0.659. progress_shaping/shaped_progress 100% active, not preventing crashes.

**Training Dynamics**: No temporal snapshots; only final-policy data.

**Signal Quality**: landing_contact_reward dead for 88% of episodes, missing soft-landing attractor. shaped_progress gate may be ineffective.

**Evidence Confidence**: `medium`
