# Subagent Research Signal

**Key Findings**: Eval score=88.1, ep_len=982 (near max 1000). Only 3/20 terminated, 17/20 truncated. Score range wide: [34, 246]. Generated reward dominates original (-0.021).

**Component Anomalies**: landing_prep dominates at 57.5% signed share (100% active), progress_gated=29.9% (73.4% active), fuel_penalty=-12.7% (100% active). Engine always firing. No dead components but landing_prep overshadows progress signal ~2:1.

**Training Dynamics**: No temporal dynamics available — no monitor snapshots captured. Cannot assess scaffold→final drift or early-vs-late activation profiles.

**Signal Quality**: Main issue: landing_prep is always active regardless of actual landing success; 17/20 episodes truncate without termination. progress_gated gate likely too restrictive (26.6% inactive). No clear attractor for engine-off behavior.

**Evidence Confidence**: `medium`
