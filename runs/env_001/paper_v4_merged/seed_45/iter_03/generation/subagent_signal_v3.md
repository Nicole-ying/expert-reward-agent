# Subagent Research Signal

**Key Findings**: score=144.8, ep_len=960.1, term_rate=5% (1/20). Contact_reward dominates with 98.7% signed share (episode sum mean=140.1). Other components negligible.

**Component Anomalies**: contact_reward >70% share (dominating). orientation_penalty and speed_penalty nearly dead (active rates 1.9%, 3.3%). progress_delta 100% active but only 1.0% share.

**Training Dynamics**: No checkpoint snapshots provided; temporal trends unknown.

**Signal Quality**: Dead gates: orientation/speed penalties rarely triggered. Contact_reward overshadows progress_delta, which provides tiny signal. Missing attractor: low termination suggests agent optimizes contact survival without successful landing.

**Evidence Confidence**: `medium`
