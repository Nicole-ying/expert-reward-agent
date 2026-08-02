# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_gated + soft_landing | -110.63 | -110.63 | 0.00 | 68.45 | progress_gated=0.002 soft_landing=0.008 | new_best |
| 2 | progress_gated + proximity_stability | 98.82 | 98.82 | 0.00 | 372.45 | progress_gated=0.001 proximity_stability=3.735 | new_best |
| 3 | fuel_penalty + progress_gated + proximity_stability | -113.41 | 98.82 | -212.22 | 68.40 | fuel_penalty=-0.017 progress_gated=0.002 proximity_stability=0.158 | no_meaningful_improvement |
| 4 | fuel_penalty + progress_gated + proximity_stability | -4.76 | 98.82 | -103.58 | 999.45 | fuel_penalty=-0.156 progress_gated=0.368 proximity_stability=5.560 | no_meaningful_improvement |
| 5 | fuel_penalty + landing_progress + progress_gated | -30.59 | 98.82 | -129.40 | 1000.00 | fuel_penalty=-0.153 landing_progress=66.073 progress_gated=0.399 | unsolved_high_achievement_continue_from_best |
| 6 | fuel_penalty + landing_prep + progress_gated | 88.06 | 98.82 | -10.75 | 982.05 | fuel_penalty=-0.136 landing_prep=1.496 progress_gated=0.313 | no_meaningful_improvement |
| 7 | fuel_penalty + landing + progress_gated | -14.66 | 98.82 | -113.47 | 1000.00 | fuel_penalty=-0.115 landing=2.101 progress_gated=0.450 | no_meaningful_improvement |
| 8 | fuel_penalty + landing + progress_delta | -114.07 | 98.82 | -212.89 | 68.40 | fuel_penalty=-0.014 landing=0.066 progress_delta=0.017 | unsolved_high_achievement_continue_from_best |
| 9 | fuel_penalty + landing + progress_delta | -109.49 | 98.82 | -208.31 | 68.45 | fuel_penalty=-0.015 landing=0.069 progress_delta=0.065 | no_meaningful_improvement |
| 10 | brake_reward + landing + progress_delta | -36.36 | 98.82 | -135.18 | 974.85 | brake_reward=0.022 landing=2.237 progress_delta=0.015 | no_meaningful_improvement |
