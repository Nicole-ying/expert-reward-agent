# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + energy_penalty + gated_forward_reward | 264.12 | 264.12 | 0.00 | 746.05 | angle_penalty=-0.002 energy_penalty=-0.022 gated_forward_reward=0.558 | new_best |
| 2 | angle_penalty + energy_penalty + gated_forward_reward | 61.77 | 264.12 | -202.35 | 438.80 | angle_penalty=-0.001 energy_penalty=-0.017 gated_forward_reward=0.330 | no_meaningful_improvement |
| 3 | angle_penalty + energy_penalty + gated_forward_reward | 213.71 | 264.12 | -50.41 | 913.00 | angle_penalty=-0.001 energy_penalty=-0.019 gated_forward_reward=0.250 | no_meaningful_improvement |
| 4 | angle_penalty + energy_penalty + gated_forward_reward | 308.64 | 308.64 | 0.00 | 798.80 | angle_penalty=-0.001 energy_penalty=-0.020 gated_forward_reward=0.248 | target_solved_new_best |
| 5 | energy_penalty + gated_forward_reward | 297.26 | 308.64 | -11.37 | 997.80 | energy_penalty=-0.018 gated_forward_reward=0.234 | stop_after_solved_drop_keep_best |
