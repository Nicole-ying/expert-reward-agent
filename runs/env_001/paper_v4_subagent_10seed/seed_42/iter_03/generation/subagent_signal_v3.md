# Subagent Research Signal

**Key Findings**: Eval score -30.3, all episodes terminate early (len 95.85). Shaped reward mean 0.028 vs original env -1.27 (misaligned). Progress_delta dominates (69% share, 94% active); soft_landing sparse (3.4% active) but contributes 27%.

**Component Anomalies**: soft_landing: very low active rate yet high share (27%), indicating large spikes. orientation_penalty: always active but negligible magnitude. progress_delta: dominant positive driver.

**Training Dynamics**: No temporal snapshots; cannot assess trends.

**Signal Quality**: No dead components. soft_landing unreliable due to sparsity. Shaped reward not aligned with true objective, likely encouraging fast progress at cost of early crashes. No attractor for stable landing.

**Evidence Confidence**: `medium`
