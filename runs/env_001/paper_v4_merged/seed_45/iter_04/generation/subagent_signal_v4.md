# Subagent Research Signal

**Key Findings**: Mean score 141.3, termination 15%, episode length 897.7. Reward dominated by contact_reward (59.9% share) and landing_progress (39.0%).

**Component Anomalies**: orientation_penalty and speed_penalty near-dead (active rates 2.6%, 4.8%, negligible shares). progress_delta active 100% but contribution trivial (0.7% share). No single component >70% but contact+landing sum to 98.9%.

**Training Dynamics**: No temporal snapshot data available; cannot assess checkpoint trends.

**Signal Quality**: Low termination rate (15%) despite high score. Speed/orientation penalties rarely trigger; thresholds may be too high. progress_delta fails to differentiate. Landing_progress multiplicative coupling, but attractor for early landing may be insufficient.

**Evidence Confidence**: `medium`
