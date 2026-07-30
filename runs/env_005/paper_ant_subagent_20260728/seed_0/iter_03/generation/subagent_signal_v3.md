# Subagent Research Signal

**Key Findings**: Score -383.1, 14/20 terminated, ep_len 503.6. Forward (+329.7/ep) and upright_penalty (-367.1/ep) are nearly equal magnitude and opposite sign — they cancel almost perfectly. Score range [-1658.2, 9.2] shows high variance with catastrophic tails.

**Component Anomalies**: upright_penalty at 47.4% signed share dominates negative side, forward at 42.5% dominates positive. These two alone account for ~90% of reward magnitude. height_penalty is near-dead (active 41%, -1.57/ep). lateral_penalty modest (-32.1/ep). No component is truly dead but forward/upright are self-cancelling.

**Training Dynamics**: No temporal snapshots available — training dynamics unknown. Final policy shows agent learned forward locomotion (+329.7/ep) but cannot maintain upright posture, suggesting the upright signal never shaped behavior effectively before forward locomotion emerged and dominated.

**Signal Quality**: The upright penalty signal is reachable (100% active rate) but ineffective: the agent experiences the penalty without escaping it. height_penalty's safe zone [0.35,0.85] is rarely violated (41% active), so its signal is weak. No early-terminal episodes — failures are mid-run collapses, not immediate falls.

**Evidence Confidence**: `medium`
