# Subagent Research Signal

**Key Findings**: Eval reward -115.5 (original env), len 131. Generated total reward/step 0.10, original reward/step -0.93. Contact success sparse (0.6% active) but 66% share when hit.

**Component Anomalies**: contact_success_reward: 66% magnitude share, 0.6% active – rare spike. progress: 16.6% magnitude vs 9.7% signed share => negative contributions.

**Training Dynamics**: No temporal snapshots provided; cannot assess component trends across checkpoints.

**Signal Quality**: Sparse contact reward; generated reward misaligned with original objective (score -115). Progress can be negative, causing self-cancellation. No attractor for safe landing.

**Evidence Confidence**: `medium`
