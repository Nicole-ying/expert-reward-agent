# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | balance_angle_penalty + balance_angular_vel_penalty + forward_progress_reward + gate_factor + gated_forward_reward | 286.89 | 286.89 | 0.00 | 1024.30 | balance_angle_penalty=-0.001 balance_angular_vel_penalty=0.000 forward_progress_reward=0.078 gate_factor=0.805 gated_forward_reward=0.068 | new_best |
| 2 | balance_angle_penalty + balance_angular_vel_penalty + forward_progress_reward + gate_factor + gated_forward_reward | 304.50 | 304.50 | 0.00 | 885.10 | balance_angle_penalty=-0.002 balance_angular_vel_penalty=0.000 forward_progress_reward=0.153 gate_factor=0.833 gated_forward_reward=0.138 | target_solved_new_best |
| 3 | balance_angle_penalty + balance_angular_vel_penalty + energy_penalty + forward_progress_reward + gate_factor + gated_forward_reward | 214.49 | 304.50 | -90.01 | 626.40 | balance_angle_penalty=-0.002 balance_angular_vel_penalty=0.000 energy_penalty=-0.054 forward_progress_reward=0.179 gate_factor=0.789 | stop_after_solved_drop_keep_best |
