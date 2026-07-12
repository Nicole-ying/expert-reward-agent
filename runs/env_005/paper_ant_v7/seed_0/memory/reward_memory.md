# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + forward_reward + height_penalty + upright_penalty | -12.04 | -12.04 | 0.00 | 80.60 | action_cost=-0.043 forward_reward=0.461 height_penalty=-0.021 upright_penalty=-2.311 | new_best |
| 2 | action_cost + forward_reward + upright_penalty | -73.71 | -12.04 | -61.67 | 731.50 | action_cost=-0.042 forward_reward=0.888 upright_penalty=-3.588 | no_meaningful_improvement |
| 3 | action_cost + forward_reward + height_reward + upright_penalty | -54.87 | -12.04 | -42.83 | 392.30 | action_cost=-0.199 forward_reward=0.271 height_reward=0.187 upright_penalty=-0.824 | no_meaningful_improvement |
| 4 | action_cost + forward_reward + height_reward + upright_penalty | 260.75 | 260.75 | 0.00 | 421.00 | action_cost=-0.191 forward_reward=0.643 height_reward=0.244 upright_penalty=-0.145 | new_best |
| 5 | action_cost + forward_reward + height_reward + upright_penalty | 407.46 | 407.46 | 0.00 | 437.70 | action_cost=-0.199 forward_reward=1.829 height_reward=0.243 upright_penalty=-0.132 | new_best |
| 6 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 662.91 | 662.91 | 0.00 | 621.95 | action_cost=-0.217 forward_reward=1.538 height_boundary_penalty=-0.001 height_reward=0.208 upright_penalty=-0.192 | new_best |
| 7 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 27.15 | 662.91 | -635.76 | 404.20 | action_cost=-0.200 forward_reward=1.102 height_boundary_penalty=-0.001 height_reward=0.225 upright_penalty=-0.691 | no_meaningful_improvement |
| 8 | action_cost + forward_reward + height_boundary_penalty + height_reward + upright_penalty | 1414.47 | 1414.47 | 0.00 | 702.90 | action_cost=-0.205 forward_reward=2.172 height_boundary_penalty=-0.001 height_reward=0.221 upright_penalty=-0.288 | new_best |
| 9 | action_cost + forward_reward + height_boundary_penalty + height_reward + lateral_velocity_penalty | 370.67 | 1414.47 | -1043.79 | 792.10 | action_cost=-0.194 forward_reward=0.579 height_boundary_penalty=-0.001 height_reward=0.173 lateral_velocity_penalty=-0.098 | no_meaningful_improvement |
| 10 | action_cost + forward_reward + height_boundary_penalty + height_reward + lateral_velocity_penalty | 10.13 | 1414.47 | -1404.34 | 380.65 | action_cost=-0.200 forward_reward=0.910 height_boundary_penalty=-0.001 height_reward=0.194 lateral_velocity_penalty=-0.028 | no_meaningful_improvement |
