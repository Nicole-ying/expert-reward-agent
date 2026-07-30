# Subagent Research Signal

**Key Findings**: Eval score=-277.9 vs generated reward +0.49/step. Episodes run full length (986.65 steps, 19/20 truncated). Agent accumulates large positive generated reward (~1490/ep) but fails external eval. Original env reward -0.95/step.

**Component Anomalies**: Gated_forward dominates at 65.7% signed share (ep_sum=1030.8). Height_penalty nearly dead: 5.5% active rate, -0.0% share (ep_sum=-0.07). The height safety mechanism is not engaging — agent is never in the z<0.3 or z>0.8 penalty regions.

**Training Dynamics**: Generated reward structure is stable per-step but disconnected from eval. All components show consistent activation profiles. No scaffold→final drift observed in composition shares. The gate threshold at 0.3 may not be constraining behavior.

**Signal Quality**: Core mismatch: generated reward is positive while eval is deeply negative. Height_penalty is a dead signal — the agent never enters the penalty zone. If height is the proxy for eval-relevant behavior, the penalty thresholds (0.3/0.8) may be unreachable or irrelevant to the task.

**Evidence Confidence**: `medium`
