# Subagent Research Signal

**Key Findings**: Eval score -336.8, 3/20 terminated. Shaped reward per-step +0.403, dominated by forward_progress (86.8% share, ep sum 854), but true env reward -0.812 per step. Gap: shaped reward fails to capture true task penalties.

**Component Anomalies**: forward_progress overwhelmingly positive, penalties (roll_pitch_hinge, lateral_drift_hinge) both negative but small magnitude (<7% share). No cancellation; all components align positively in shaped sum, yet true reward negative.

**Training Dynamics**: No temporal snapshots provided; only final policy stats available. Trend data absent.

**Signal Quality**: Gates (height, upright) allow forward_progress in tilted or boundary states, leading to shaped reward positivity while true reward suffers. Lenient thresholds (e.g., upright full at 0.5, ang_vel margin 1.0) likely contribute. Dead gates not observed but high false positive signal.

**Evidence Confidence**: `medium`
