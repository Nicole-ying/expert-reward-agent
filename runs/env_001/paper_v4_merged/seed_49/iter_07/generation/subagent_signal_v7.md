# Subagent Research Signal

**Key Findings**: Mean eval reward 134.35, 1/20 terminated. Progress dominates (98.5% magnitude share). Overspeed penalty nearly dead (3.7% active).

**Component Anomalies**: overspeed_penalty nearly dead (active_rate 3.7%, signed_share -1.5%). Progress is sole active component (100% active).

**Training Dynamics**: No temporal checkpoint data; dynamics across training unknown.

**Signal Quality**: overspeed_penalty threshold rarely crossed (3.7% active). Low termination rate (5%) suggests missing attractor for precise goal completion.

**Evidence Confidence**: `medium`
