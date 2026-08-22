# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward + height_penalty + upright_penalty | -11.32 | -11.32 | 0.00 | 59.15 | forward=0.617 height_penalty=-0.003 upright_penalty=-1.383 | new_best |
| 2 | forward_gated + health_gate + upright_penalty + z_velocity_penalty | -240.56 | -11.32 | -229.24 | 244.40 | forward_gated=0.259 health_gate=0.515 upright_penalty=-0.367 z_velocity_penalty=-0.004 | no_meaningful_improvement |
| 3 | forward_gated + health_gate + upright_penalty + z_velocity_penalty | -101.97 | -11.32 | -90.65 | 319.60 | forward_gated=0.374 health_gate=0.574 upright_penalty=-0.323 z_velocity_penalty=-0.008 | no_meaningful_improvement |
| 4 | forward + height_factor + upright_bonus | 69.24 | 69.24 | 0.00 | 989.80 | forward=0.641 height_factor=0.874 upright_bonus=0.028 | unsolved_improving_continue_from_best |
| 5 | forward + height_factor + lateral_penalty + upright_bonus + vertical_penalty | -664.53 | 69.24 | -733.77 | 713.20 | forward=0.182 height_factor=0.771 lateral_penalty=-0.017 upright_bonus=-0.073 vertical_penalty=-0.011 | no_meaningful_improvement |
| 6 | forward + height_factor + lateral_penalty + upright_penalty + vertical_penalty | -231.68 | 69.24 | -300.93 | 320.35 | forward=0.650 height_factor=0.849 lateral_penalty=-0.033 upright_penalty=-0.245 vertical_penalty=-0.020 | no_meaningful_improvement |
