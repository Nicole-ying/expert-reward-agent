# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | 0.72 | 0.72 | 0.00 | 11.80 | action_penalty=-0.003 forward=0.222 height_penalty=-0.032 lateral_penalty=-0.097 upright_penalty=-5.673 | new_best |
| 2 | action_penalty + forward + height_penalty + lateral_penalty + upright_penalty | -383.09 | 0.72 | -383.81 | 503.55 | action_penalty=-0.003 forward=0.372 height_penalty=-0.003 lateral_penalty=-0.117 upright_penalty=-0.758 | no_meaningful_improvement |
| 3 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | 1839.71 | 1839.71 | 0.00 | 981.50 | action_penalty=-0.003 gated_forward=1.150 height_penalty=-0.000 lateral_penalty=-0.103 upright_bonus=0.095 | new_best |
| 4 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -591.52 | 1839.71 | -2431.23 | 585.90 | action_penalty=-0.003 gated_forward=0.141 height_penalty=-0.001 lateral_penalty=-0.015 upright_bonus=-0.011 | no_meaningful_improvement |
| 5 | action_penalty + ang_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -73.65 | 1839.71 | -1913.36 | 700.30 | action_penalty=-0.003 ang_penalty=-0.081 gated_forward=0.647 height_penalty=-0.000 lateral_penalty=-0.024 | no_meaningful_improvement |
| 6 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -442.53 | 1839.71 | -2282.23 | 837.25 | action_penalty=-0.003 gated_forward=0.428 height_penalty=-0.000 lateral_penalty=-0.020 upright_bonus=0.114 | unsolved_high_achievement_continue_from_best |
| 7 | action_penalty + gated_forward + height_penalty + lateral_penalty + upright_bonus | -277.90 | 1839.71 | -2117.61 | 986.65 | action_penalty=-0.003 gated_forward=0.361 height_penalty=-0.000 lateral_penalty=-0.060 upright_bonus=0.194 | no_meaningful_improvement |
| 8 | _height_gate + action_penalty + forward_reward + lateral_penalty + upright_penalty | 464.77 | 1839.71 | -1374.93 | 907.70 | _height_gate=0.729 action_penalty=-0.003 forward_reward=0.690 lateral_penalty=-0.161 upright_penalty=-0.259 | no_meaningful_improvement |
| 9 | _height_gate + action_penalty + forward_reward + lateral_penalty + upright_penalty | 424.60 | 1839.71 | -1415.11 | 872.50 | _height_gate=0.702 action_penalty=-0.003 forward_reward=0.685 lateral_penalty=-0.106 upright_penalty=0.000 | unsolved_high_achievement_continue_from_best |
| 10 | _height_gate + action_penalty + forward_reward + lateral_penalty + upright_bonus | -292.47 | 1839.71 | -2132.17 | 785.65 | _height_gate=0.680 action_penalty=-0.003 forward_reward=0.489 lateral_penalty=-0.093 upright_bonus=0.122 | no_meaningful_improvement |
