# Subagent Research Signal

**Key Findings**: score=-23.46, len=1000 (all truncated). approach component dom: episode sum mean=537, signed share=100%. soft_landing dead (active rate 0%). orientation penalty active 100% but negligible magnitude.

**Component Anomalies**: soft_landing: dead gate (active 0%). approach: dominates (>100% signed share due to negative orientation share being tiny).

**Training Dynamics**: no temporal data; final policy snapshot only.

**Signal Quality**: soft_landing proximity gate (<0.3) never crossed. approach signal large, active 88.4% steps, lacking counterbalance for landing.

**Evidence Confidence**: `low`
