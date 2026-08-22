# Subagent Research Signal

**Key Findings**: mean_eval_reward=-114.97, all 20 episodes early-terminated at mean len=68.4. proximity_delta dominates at +55.9/episode (83.2% signed share). landing_bonus is completely dead: 0.0% active rate, 0.0 magnitude share.

**Component Anomalies**: landing_bonus is a phantom component — present in code (w_land=80) but contributes zero signal (active_rate=0.0%). All learning relies on proximity_delta, which encourages approaching origin but not soft landing.

**Mechanism Hypothesis**: The landing_bonus's multi-factor gating (dist_factor=e^(-dist/0.3), velocity/angle factors zero above |v|>0.3 or |angle|>0.3) is too strict: the lander never simultaneously satisfies distance, velocity, and angle conditions at leg contact, so the bonus never fires and cannot guide soft-landing behavior.

**Decision Implication**: PATCH landing_bonus: relax thresholds (e.g., dist_factor sigma from 0.3→1.5, velocity/angle cutoffs from 0.3→1.0) so the component activates and provides a usable landing gradient. Keep other components unchanged.

**Confidence**: `high`
