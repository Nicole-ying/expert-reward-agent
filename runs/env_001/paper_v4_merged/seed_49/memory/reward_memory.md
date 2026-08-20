# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -113.71 | -113.71 | 0.00 | 68.45 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.016 soft_landing=0.012 | new_best |
| 2 | angle_penalty + angvel_penalty + efficiency + progress + soft_landing | -115.68 | -113.71 | -1.96 | 68.35 | angle_penalty=-0.001 angvel_penalty=-0.003 efficiency=-0.003 progress=0.003 soft_landing=0.012 | no_meaningful_improvement |
| 3 | angle_penalty + angvel_penalty + efficiency + failure_penalty + progress + soft_landing | -222.06 | -113.71 | -108.35 | 148.50 | angle_penalty=-0.011 angvel_penalty=-0.000 efficiency=-0.013 failure_penalty=-0.021 progress=-0.003 | no_meaningful_improvement |
| 4 | angle_penalty + efficiency + failure_penalty + progress + success_bonus | -120.20 | -113.71 | -6.49 | 68.35 | angle_penalty=-0.001 efficiency=-0.009 failure_penalty=-0.004 progress=0.009 success_bonus=0.059 | unsolved_stagnation_fresh_restart |
| 5 | angle_penalty + angvel_penalty + progress | -53.83 | -53.83 | 0.00 | 100.40 | angle_penalty=-0.001 angvel_penalty=-0.000 progress=0.013 | new_best |
| 6 | overspeed_penalty + progress | 134.35 | 134.35 | 0.00 | 956.10 | overspeed_penalty=-0.000 progress=0.004 | new_best |
| 7 | overspeed_penalty + progress + success_bonus | -114.07 | 134.35 | -248.42 | 1000.00 | overspeed_penalty=-0.000 progress=0.005 success_bonus=68.061 | no_meaningful_improvement |
