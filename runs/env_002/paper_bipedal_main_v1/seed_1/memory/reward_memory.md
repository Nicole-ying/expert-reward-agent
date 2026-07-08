# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + stability_penalty | 280.50 | 280.50 | 0.00 | 908.15 | forward_reward=0.335 stability_penalty=-0.061 | new_best |
| 2 | energy_penalty + forward_reward + stability_penalty | 313.46 | 313.46 | 0.00 | 1158.55 | energy_penalty=-0.060 forward_reward=0.353 stability_penalty=-0.071 | target_solved_new_best |
| 3 | energy_penalty + forward_reward + gait_reward + stability_penalty | 313.73 | 313.73 | 0.00 | 1087.90 | energy_penalty=-0.071 forward_reward=0.343 gait_reward=0.018 stability_penalty=-0.065 | target_solved_new_best |
| 4 | energy_penalty + forward_reward + gait_reward + stability_penalty | 294.44 | 313.73 | -19.30 | 1084.10 | energy_penalty=-0.071 forward_reward=0.306 gait_reward=0.016 stability_penalty=-0.110 | stop_after_solved_drop_keep_best |
