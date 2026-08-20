# Subagent Research Signal

**Key Findings**: score=128.47, len=1000, 0 terminations. Reward components per-step means near zero (progress_delta=0.0023). Episode sum of progress_delta=1.40 dominates signed share (91.1%) but is small relative to total score.

**Component Anomalies**: penalties (orientation, speed) active <10% of steps (final policy: orientation 0.6%, speed 1.0%). Their magnitude shares negligible. No dead components, but extremely low activity.

**Training Dynamics**: No temporal monitor snapshots; dynamics across checkpoints unknown.

**Signal Quality**: Custom reward signals are dwarfed by an unexplained base reward (score=128). Penalties rarely triggered, thresholds likely too high relative to policy behavior. Missing attractor for shaping desired behavior.

**Evidence Confidence**: `low`
