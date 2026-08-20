# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + forward_gated | 2.91 | 2.91 | 0.00 | 694.35 | action_penalty=-0.002 forward_gated=0.425 | new_best |
| 2 | action_penalty + forward_gated + lateral_penalty | -416.10 | 2.91 | -419.01 | 625.10 | action_penalty=-0.043 forward_gated=0.145 lateral_penalty=-0.051 | no_meaningful_improvement |
| 3 | action_penalty + forward_gated + lateral_penalty | 13.55 | 13.55 | 0.00 | 740.65 | action_penalty=-0.042 forward_gated=0.307 lateral_penalty=-0.058 | new_best |
| 4 | action_penalty + forward_gated + lateral_penalty | -464.62 | 13.55 | -478.17 | 583.05 | action_penalty=-0.044 forward_gated=0.211 lateral_penalty=-0.055 | unsolved_stagnation_fresh_restart |
| 5 | forward_progress + lateral_penalty | -1815.60 | 13.55 | -1829.15 | 1000.00 | forward_progress=0.285 lateral_penalty=-0.101 | no_meaningful_improvement |
| 6 | forward_progress + lateral_penalty | -174.24 | 13.55 | -187.78 | 435.00 | forward_progress=0.912 lateral_penalty=-0.170 | no_meaningful_improvement |
| 7 | action_penalty + health_gate + lateral_penalty + progress | -381.25 | 13.55 | -394.80 | 677.50 | action_penalty=-0.083 health_gate=0.299 lateral_penalty=-0.038 progress=0.589 | unsolved_stagnation_fresh_restart |
| 8 | forward_speed_reward + height_penalty + upright_penalty | -311.98 | 13.55 | -325.53 | 138.95 | forward_speed_reward=0.891 height_penalty=-0.000 upright_penalty=-0.086 | no_meaningful_improvement |
| 9 | forward_speed_reward + height_factor + height_penalty + upright_factor + upright_penalty | -661.90 | 13.55 | -675.45 | 473.80 | forward_speed_reward=0.246 height_factor=0.712 height_penalty=0.000 upright_factor=0.436 upright_penalty=0.000 | no_meaningful_improvement |
| 10 | forward_speed_reward + height_factor + survival_reward + upright_factor | -254.40 | 13.55 | -267.95 | 304.60 | forward_speed_reward=0.233 height_factor=0.752 survival_reward=0.041 upright_factor=0.462 | unsolved_stagnation_fresh_restart |
