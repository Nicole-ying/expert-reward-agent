# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | balance_penalty + forward_progress | -59.97 | -59.97 | 0.00 | 246.25 | balance_penalty=-0.011 forward_progress=0.185 | new_best |
| 2 | air_stability_penalty + balance_penalty + forward_progress | -86.30 | -59.97 | -26.33 | 105.95 | air_stability_penalty=-0.142 balance_penalty=-0.011 forward_progress=0.215 | no_meaningful_improvement |
| 3 | air_stability_penalty + balance_penalty + forward_reward + terrain_gate + terrain_roughness | -95.84 | -59.97 | -35.87 | 74.80 | air_stability_penalty=-0.075 balance_penalty=-0.005 forward_reward=0.072 terrain_gate=0.501 terrain_roughness=0.214 | no_meaningful_improvement |
| 4 | balance_penalty + forward_reward + terrain_gate + terrain_roughness | -74.85 | -59.97 | -14.88 | 376.95 | balance_penalty=-0.008 forward_reward=0.106 terrain_gate=0.497 terrain_roughness=0.216 | unsolved_stagnation_fresh_restart |
| 5 | energy_penalty + forward_reward + hinge_penalty | -52.46 | -52.46 | 0.00 | 243.15 | energy_penalty=-0.018 forward_reward=0.183 hinge_penalty=-0.001 | new_best |
| 6 | energy_penalty + forward_reward | -59.50 | -52.46 | -7.04 | 380.95 | energy_penalty=-0.018 forward_reward=0.175 | no_meaningful_improvement |
| 7 | energy_penalty + forward_reward + hinge_penalty | -52.19 | -52.19 | 0.00 | 401.70 | energy_penalty=-0.018 forward_reward=0.170 hinge_penalty=-0.004 | unsolved_stagnation_fresh_restart |
| 8 | action_penalty + progress | -63.34 | -52.19 | -11.15 | 217.05 | action_penalty=-0.017 progress=0.187 | no_meaningful_improvement |
| 9 | action_penalty + hinge_balance_penalty + progress | -53.58 | -52.19 | -1.39 | 216.15 | action_penalty=-0.018 hinge_balance_penalty=-0.003 progress=0.182 | no_meaningful_improvement |
| 10 | action_penalty + progress + vertical_hinge_penalty | -54.31 | -52.19 | -2.12 | 228.85 | action_penalty=-0.017 progress=0.186 vertical_hinge_penalty=-0.000 | unsolved_stagnation_fresh_restart |
