# Subagent Research Signal

**Key Findings**: Mean eval reward -24.0, terminated 1/20, len 980.8. Reward dominated by gate_angle (916.9,65.1%) and contact_factor (398.4,28.3%); progress negligible (1.19,0.1%). Agent survives but fails task.

**Component Anomalies**: gate_angle+contact_factor >93% share, over-incentivize posture. progress/shaped_progress dead (<0.1%).

**Training Dynamics**: No checkpoint data; final policy exploits angle/contact without target approach.

**Signal Quality**: shaped_progress collapsed (tiny progress); success_bonus (75.4 sum) insufficient. No dead gates but reward fails to guide to goal.

**Evidence Confidence**: `medium`
