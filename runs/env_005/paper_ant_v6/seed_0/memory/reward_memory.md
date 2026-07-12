# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + height_penalty + lateral_penalty + posture_penalty | -2.50 | -2.50 | 0.00 | 15.70 | forward_reward=0.632 height_penalty=-0.026 lateral_penalty=-0.340 posture_penalty=-1.948 | new_best |
| 2 | forward_reward + height_penalty + lateral_penalty | -14.41 | -2.50 | -11.90 | 938.85 | forward_reward=0.552 height_penalty=-0.001 lateral_penalty=-0.093 | no_meaningful_improvement |
| 3 | action_penalty + forward_reward + height_penalty + lateral_penalty | 18.41 | 18.41 | 0.00 | 20.85 | action_penalty=-0.278 forward_reward=0.336 height_penalty=-0.012 lateral_penalty=-0.157 | new_best |
| 4 | action_penalty + forward_reward + lateral_penalty | 8.92 | 18.41 | -9.50 | 17.45 | action_penalty=-0.311 forward_reward=0.333 lateral_penalty=-0.181 | unsolved_improving_continue_from_best |
| 5 | action_penalty + forward_reward + height_penalty + lateral_penalty | 2.52 | 18.41 | -15.89 | 13.75 | action_penalty=-0.323 forward_reward=0.298 height_penalty=-0.032 lateral_penalty=-0.218 | no_meaningful_improvement |
| 6 | action_penalty + descend_penalty + forward_reward + lateral_penalty | 5.70 | 18.41 | -12.71 | 11.45 | action_penalty=-0.328 descend_penalty=-0.036 forward_reward=0.259 lateral_penalty=-0.184 | no_meaningful_improvement |
| 7 | action_penalty + descend_penalty + forward_reward + height_penalty + lateral_penalty | 4.44 | 18.41 | -13.97 | 17.05 | action_penalty=-0.333 descend_penalty=-0.034 forward_reward=0.244 height_penalty=-0.014 lateral_penalty=-0.168 | unsolved_improving_continue_from_best |
