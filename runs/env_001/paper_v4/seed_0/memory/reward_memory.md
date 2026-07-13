# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + landing_reward + proximity | -70.35 | -70.35 | 0.00 | 1000.00 | fuel_penalty=-0.041 landing_reward=0.295 proximity=1.568 | new_best |
| 2 | fuel_penalty + landing_reward + progress | -129.32 | -70.35 | -58.97 | 69.65 | fuel_penalty=-0.003 landing_reward=0.002 progress=0.080 | no_meaningful_improvement |
| 3 | fuel_penalty + landing_reward + proximity | -26.30 | -26.30 | 0.00 | 951.00 | fuel_penalty=-0.034 landing_reward=2.216 proximity=0.072 | new_best |
| 4 | fuel_penalty + landing_reward + proximity | -120.27 | -26.30 | -93.97 | 68.35 | fuel_penalty=-0.006 landing_reward=0.087 proximity=0.053 | no_meaningful_improvement |
| 5 | approach_improvement + contact_event + fuel_penalty + progress_delta | -123.86 | -26.30 | -97.57 | 68.30 | approach_improvement=0.028 contact_event=0.022 fuel_penalty=-0.006 progress_delta=0.016 | no_meaningful_improvement |
| 6 | descent_safety + fuel_penalty + safe_proximity + stable_landed + touchdown_bonus | -18.28 | -18.28 | 0.00 | 911.60 | descent_safety=-0.114 fuel_penalty=-0.040 safe_proximity=0.502 stable_landed=0.003 touchdown_bonus=0.004 | new_best |
| 7 | approach_shaping + descent_safety + fuel_penalty + stable_landed + touchdown_bonus | -388.67 | -18.28 | -370.39 | 109.50 | approach_shaping=-0.016 descent_safety=-0.078 fuel_penalty=-0.040 stable_landed=0.001 touchdown_bonus=0.001 | no_meaningful_improvement |
| 8 | descent_safety + fuel_penalty + landing_proximity + stable_landed + touchdown_bonus | 224.21 | 224.21 | 0.00 | 437.10 | descent_safety=-0.218 fuel_penalty=-0.037 landing_proximity=1.030 stable_landed=0.350 touchdown_bonus=0.219 | target_solved_new_best |
| 9 | descent_safety + fuel_penalty + landing_improvement + stable_landed + touchdown_bonus | -611.76 | 224.21 | -835.97 | 146.90 | descent_safety=-0.060 fuel_penalty=-0.040 landing_improvement=0.017 stable_landed=0.000 touchdown_bonus=0.000 | stop_after_solved_drop_keep_best |
