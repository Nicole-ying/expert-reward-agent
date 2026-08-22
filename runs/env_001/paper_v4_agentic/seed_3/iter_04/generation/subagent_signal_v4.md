# Subagent Research Signal

**Key Findings**: Score=170.64 over 20 episodes, all terminated (no truncations), mean ep_len=363.3. Generated reward per step is small (mean 0.54) but accumulates; original env reward is negative (-0.47/step). Total episode reward sums are moderate positive.

**Component Anomalies**: landing_quality dominates at 45.6% signed share but fires only 12.3% of steps — sparse, high-magnitude spikes. attitude_penalty fires 100% of steps but contributes only -7.9% (constant drag, not diagnostic). progress is the only reliable continuous signal: 97.3% active, 24.9% share.

**Training Dynamics**: No temporal dynamics available — component_dynamics returned no monitor snapshots. Cannot assess scaffold→final drift, early vs late activation profiles, or checkpoint-level trends.

**Signal Quality**: landing_quality sparsity (12.3% active) suggests a threshold or gate rarely crossed — potential reachability gap. landing_velocity_penalty similarly sparse (13.9%). attitude_penalty's 100% active rate makes it a constant offset, not a behavioral signal. No coupling analysis possible without dynamics.

**Evidence Confidence**: `medium`
