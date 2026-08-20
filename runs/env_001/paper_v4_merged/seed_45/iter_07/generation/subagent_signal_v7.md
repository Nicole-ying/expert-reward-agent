# Subagent Research Signal

**Key Findings**: Score=-111.4, all episodes early-terminated. Progress (5.6) and speed penalty (-4.6) dominate and nearly cancel. Completion proxy active 0.6%.

**Component Anomalies**: Progress and speed are >40% signed shares, opposite signs, self-cancelling. angle, angvel, completion nearly dead.

**Training Dynamics**: No checkpoint data; final policy only.

**Signal Quality**: Dead: angle (3.1% active), angvel (1.6%), completion (0.6%). Speed penalty always on, creating cancellation. No attractor for successful landing.

**Evidence Confidence**: `low`
