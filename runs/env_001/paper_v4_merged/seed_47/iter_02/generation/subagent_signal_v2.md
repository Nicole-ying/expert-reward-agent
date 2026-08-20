# Subagent Research Signal

**Key Findings**: Mean eval reward -36.03, 0/20 terminations (all truncated at 1000 steps). Success_reward dead (active_rate 0%, episode_sum 0.0). Progress_reward (ep sum 1.90) and attitude_penalty (ep sum -1.18) always active but insufficient; original env reward per-step mean -0.19 driving overall negative.

**Component Anomalies**: success_reward dead (0% active). No component exceeds 70% magnitude share (progress_reward 69.7% near threshold). Original env reward not in composition table but large negative influence.

**Training Dynamics**: No monitor snapshots provided; temporal dynamics missing.

**Signal Quality**: Dead success_reward, continuous progress/attitude signals fail to correlate with task success; no termination attractor.

**Evidence Confidence**: `medium`
