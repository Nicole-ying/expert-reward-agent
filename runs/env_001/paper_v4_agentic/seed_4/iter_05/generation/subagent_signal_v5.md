# Subagent Research Signal

**Key Findings**: mean_eval_reward=-4.76, only 1/20 terminated, ep_len=999.45 (max). Generated reward 5.77/step is completely decoupled from true task score. Policy learned to exploit proximity_stability (88.6% signed share, 4364.72 ep sum) while ignoring progress_gated (7.3%, 361.38).

**Component Anomalies**: proximity_stability dominates at 88.6% signed share vs progress_gated at 7.3%. fuel_penalty is -4.1% but 100% active. The proximity component has w=10.0 with gate_min_stab=0.2 floor, ensuring a persistent nonzero signal even in poor states, while progress_gated uses w=8.0 with gate_min=0.1.

**Training Dynamics**: No component_dynamics snapshots available. Final-policy state shows the agent converged to a local optimum: maximize proximity_stability by staying near origin with low velocity/angle, achieving ~5.77/step generated reward while making zero task progress (original_env_reward=-0.012/step).

**Signal Quality**: proximity_stability has structural advantages: higher weight (10 vs 8), higher gate floor (0.2 vs 0.1), and contact_mult (+50% when legs touch). These combine to make it the path of least resistance — the policy found a stationary, low-energy posture that scores well on proximity_stability without needing to actually reach the target.

**Evidence Confidence**: `high`
