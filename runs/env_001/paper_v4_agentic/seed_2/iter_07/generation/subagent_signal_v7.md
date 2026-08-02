# Subagent Research Signal

**Key Findings**: Score=135.20, ep_len=917.45, only 2/20 terminated. A_progress_gated is functionally dead: per-step mean=0.0008 (episode sum=0.55, 0.3% share). C_landing_steady dominates at 99.7% share (episode sum=183.84). Original env reward per-step is 0.0491 vs generated 0.1377.

**Component Anomalies**: A_progress_gated is effectively silenced despite 100% activation rate — the speed_gate is attenuating the progress signal to near-zero (0.0008/step). C_landing_steady dominates at 99.7% share and 74.7% active rate, but it's a terminal-state attractor that only rewards behavior after the agent is already at the pad.

**Training Dynamics**: No temporal snapshots available from this run. Cannot confirm early-vs-late trends or scaffold→final drift. The final policy composition shows the end-state: A_progress_gated at 0.3% share indicates the gate never relaxed across training.

**Signal Quality**: CRITICAL: The only shaping signal (A_progress) that could guide the agent toward the landing pad is being crushed by the speed gate. The agent has almost no gradient to follow — C_landing_steady is a reward for being at the goal, not a signal for getting there. This is a classic missing-attractor problem: the reward landscape is flat everywhere except at the target state.

**Evidence Confidence**: `medium`
