# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + height_reward + upright_reward | 67.71 | 67.71 | 0.00 | 201.55 | forward_reward=0.668 height_reward=-0.034 upright_reward=-0.215 | new_best |
| 2 | forward_gated + height_reward | -271.10 | 67.71 | -338.81 | 463.80 | forward_gated=0.343 height_reward=-0.044 | no_meaningful_improvement |
| 3 | action_penalty + forward_gated + height_reward | -112.41 | 67.71 | -180.12 | 369.00 | action_penalty=-0.044 forward_gated=0.505 height_reward=-0.037 | no_meaningful_improvement |
| 4 | action_penalty + forward_gated_height | -55.54 | 67.71 | -123.25 | 724.55 | action_penalty=-0.043 forward_gated_height=0.462 | unsolved_stagnation_fresh_restart |
| 5 | forward_reward + height_penalty + lateral_penalty + upright_penalty | -5.09 | 67.71 | -72.80 | 39.05 | forward_reward=0.491 height_penalty=-0.022 lateral_penalty=-0.359 upright_penalty=-3.674 | no_meaningful_improvement |
| 6 | gated_forward + height_gate + lateral_penalty + upright_reward | -37.13 | 67.71 | -104.84 | 1000.00 | gated_forward=1.150 height_gate=0.817 lateral_penalty=-0.098 upright_reward=0.029 | no_meaningful_improvement |
| 7 | gated_forward + height_gate + joint_vel_penalty + lateral_penalty + upright_reward | -353.94 | 67.71 | -421.65 | 1000.00 | gated_forward=0.959 height_gate=0.840 joint_vel_penalty=-0.372 lateral_penalty=-0.051 upright_reward=0.039 | unsolved_stagnation_fresh_restart |
| 8 | forward_reward + height_penalty + lateral_penalty + upright_penalty | -12.66 | 67.71 | -80.37 | 13.25 | forward_reward=0.227 height_penalty=0.009 lateral_penalty=0.288 upright_penalty=1.278 | no_meaningful_improvement |
| 9 | gated_forward + lateral_penalty + upright_penalty | -15.71 | 67.71 | -83.42 | 9.95 | gated_forward=0.172 lateral_penalty=0.302 upright_penalty=1.319 | no_meaningful_improvement |
| 10 | gated_forward + lateral_gate + upright_penalty | -9.29 | 67.71 | -77.00 | 15.05 | gated_forward=0.081 lateral_gate=0.404 upright_penalty=0.911 | unsolved_stagnation_fresh_restart |
