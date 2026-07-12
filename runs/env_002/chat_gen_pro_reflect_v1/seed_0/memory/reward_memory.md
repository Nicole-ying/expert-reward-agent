# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gated_forward_reward + health_gate | 231.24 | 231.24 | 0.00 | 794.20 | angle_penalty=-0.004 angular_vel_penalty=-0.000 balance_penalty=-0.005 forward_reward=0.275 gated_forward_reward=0.243 | new_best |
| 2 | angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gated_forward_reward + health_gate | 306.04 | 306.04 | 0.00 | 818.00 | angle_penalty=-0.010 angular_vel_penalty=-0.000 balance_penalty=-0.010 forward_reward=0.266 gated_forward_reward=0.236 | target_solved_new_best |
| 3 | action_penalty + angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gated_forward_reward | 329.81 | 329.81 | 0.00 | 865.00 | action_penalty=-0.048 angle_penalty=-0.010 angular_vel_penalty=-0.000 balance_penalty=-0.010 forward_reward=0.288 | target_solved_new_best |
| 4 | action_penalty + angle_penalty + angular_vel_penalty + balance_penalty + forward_reward + gated_forward_reward | 283.87 | 329.81 | -45.95 | 906.40 | action_penalty=-0.051 angle_penalty=-0.010 angular_vel_penalty=-0.000 balance_penalty=-0.010 forward_reward=0.328 | stop_after_solved_drop_keep_best |
