# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_upright_product + height_survival_penalty | -544.72 | -544.72 | 0.00 | 647.80 | forward_upright_product=0.253 height_survival_penalty=-0.023 | new_best |
| 2 | forward_upright_product + height_survival_penalty + torque_penalty | -49.90 | -49.90 | 0.00 | 349.35 | forward_upright_product=0.447 height_survival_penalty=-0.017 torque_penalty=-0.016 | new_best |
| 3 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -9.56 | -9.56 | 0.00 | 16.35 | forward_upright_gated=0.125 height_gate_value=0.450 lateral_penalty=-0.105 rotation_penalty=-0.260 torque_penalty=-0.018 | new_best |
| 4 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -779.13 | -9.56 | -769.57 | 797.25 | forward_upright_gated=0.154 height_gate_value=0.498 lateral_penalty=-0.023 rotation_penalty=-0.068 torque_penalty=-0.018 | no_meaningful_improvement |
| 5 | forward_upright_gated + height_gate_value + lateral_penalty + rotation_penalty + torque_penalty | -337.63 | -9.56 | -328.07 | 493.45 | forward_upright_gated=1.178 height_gate_value=0.395 lateral_penalty=-0.027 rotation_penalty=-0.096 torque_penalty=-0.017 | unsolved_stagnation_fresh_restart |
| 6 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -184.03 | -9.56 | -174.48 | 577.30 | angular_velocity_penalty=-0.128 gated_forward=0.751 height_hinge_penalty=-0.151 lateral_penalty=-0.177 | no_meaningful_improvement |
| 7 | angular_velocity_penalty + gated_forward + height_hinge_penalty + lateral_penalty | -282.04 | -9.56 | -272.48 | 663.90 | angular_velocity_penalty=-0.114 gated_forward=0.600 height_hinge_penalty=-0.185 lateral_penalty=-0.143 | no_meaningful_improvement |
| 8 | forward_reward + height_hinge_penalty + upright_reward | -6.63 | -6.63 | 0.00 | 616.50 | forward_reward=1.271 height_hinge_penalty=-0.009 upright_reward=0.179 | unsolved_stagnation_fresh_restart |
| 9 | angular_penalty + forward_reward + lateral_penalty | -388.82 | -6.63 | -382.19 | 702.30 | angular_penalty=-0.172 forward_reward=0.378 lateral_penalty=-0.021 | no_meaningful_improvement |
| 10 | angular_penalty + forward_reward + lateral_penalty | -169.50 | -6.63 | -162.86 | 839.35 | angular_penalty=-0.235 forward_reward=0.609 lateral_penalty=-0.035 | no_meaningful_improvement |
