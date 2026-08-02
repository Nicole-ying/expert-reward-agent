# Subagent Research Signal

**Key Findings**: Eval score = -90.2, episodes run full 1000 steps (0/20 terminated), landing_bonus never fires (active 0%).

**Component Anomalies**: landing_bonus dead (0% active, 0% share); progress dominates (91% magnitude share) but yields no termination.

**Training Dynamics**: No temporal data; final policy shows no early termination, indicating no progression toward successful landing.

**Signal Quality**: Missing attractor: landing_bonus gate (v<0.2, contact) never activates, so no positive signal for soft landing.

**Evidence Confidence**: `medium`
