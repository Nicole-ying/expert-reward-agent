# Subagent Research Signal

**Key Findings**: Score=-111.37, all episodes terminated but 80% early (<150 steps, score<-50). Shaped reward per-step positive (0.14) but original env reward negative (-1.24). Progress dominates (50.5% signed share), landing bonus almost never active (0.3% active rate).

**Component Anomalies**: landing_bonus dead: active rate 0.3%, episode sum mean 3.5 from sporadic triggers. Progress has 100% nonzero, dominates magnitude share (65.6%). Stability and lateral drift penalties always active but low magnitude shares (11.4%, 8.8%).

**Training Dynamics**: No monitor snapshots; no temporal trend data available across checkpoints.

**Signal Quality**: Dead gate: landing_bonus rarely meets thresholds (both legs contact, |angle|<0.1, |vy|<0.2). Progress always positive, failing to distinguish successful landing from uncontrolled descent. Missing attractor for soft landing behavior. No coupling analysis possible from this data.

**Evidence Confidence**: `medium`
