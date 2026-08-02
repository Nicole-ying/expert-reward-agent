# Subagent Research Signal

**Key Findings**: Score=-122, all 20 episodes early terminal. progress_shaping dominates (77.5% signed share, 100% active) yet unable to prevent early crashes.

**Component Anomalies**: progress_shaping >70% share. landing_contact_reward high magnitude (14.5%) but sparse (3.1% active). angle_hinge dead (0.1% active, ~0 reward).

**Training Dynamics**: No temporal data; only final policy. Drift undetectable from single snapshot.

**Signal Quality**: angle_hinge dead gate. landing_contact_reward sparse, insufficient feedback. Missing attractor for successful landing, causing universal early termination.

**Evidence Confidence**: `medium`
