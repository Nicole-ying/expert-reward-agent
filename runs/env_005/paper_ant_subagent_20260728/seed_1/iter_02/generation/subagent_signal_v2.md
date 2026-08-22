# Subagent Research Signal

**Key Findings**: Score=-11.32, all 20 episodes terminated early (mean len=59.2 steps). Forward reward dominates magnitude share at 92.8% (ep sum ~72.3), while upright penalty is 100% active but only -6.9% signed share (ep sum ~-5.4). Generated and original env rewards are nearly identical (~-0.77/step).

**Component Anomalies**: Height penalty is dead: 2.5% active rate, 0.2% magnitude share, ep sum ~-0.17. Upright penalty fires on 100% of steps — no differentiation between good/bad posture, acts as a constant negative bias rather than a behavioral signal. Forward reward is the only meaningful component but insufficient to offset poor original env reward.

**Training Dynamics**: No temporal snapshots available. Cannot assess early vs late training drift or scaffold evolution. Only final-policy composition is observable.

**Signal Quality**: Height penalty threshold likely unreachable under current policy — 2.5% activation means the gate almost never opens. Upright penalty has no dynamic range (100% active, constant magnitude), so it cannot guide learning toward better posture. Both penalty components lack an attractor: the agent gets penalized regardless of behavior, providing no gradient toward the desired outcome.

**Evidence Confidence**: `medium`
