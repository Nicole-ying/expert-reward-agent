# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + angvel_penalty + forward_reward + vert_penalty | 212.37 | 212.37 | 0.00 | 632.50 | angle_penalty=-0.004 angvel_penalty=-0.000 forward_reward=0.358 vert_penalty=0.000 | new_best |
| 2 | angvel_penalty + forward_reward + gated_forward_reward + stability_gate + vert_penalty | 316.49 | 316.49 | 0.00 | 892.95 | angvel_penalty=-0.000 forward_reward=0.346 gated_forward_reward=0.328 stability_gate=0.921 vert_penalty=0.000 | target_solved_new_best |
| 3 | angvel_penalty + gait_cycle_reward + tilt_penalty + torque_penalty | -92.75 | 316.49 | -409.25 | 1600.00 | angvel_penalty=-0.000 gait_cycle_reward=0.133 tilt_penalty=-0.001 torque_penalty=-0.001 | stop_after_solved_drop_keep_best |
