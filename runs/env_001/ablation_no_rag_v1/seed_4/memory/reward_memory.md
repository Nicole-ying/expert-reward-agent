# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angvel_penalty + orientation_penalty + target_proximity + velocity_penalty | -53.29 | -53.29 | 0.00 | 72.70 | angvel_penalty=-0.011 orientation_penalty=-0.001 target_proximity=-0.212 velocity_penalty=-0.063 | new_best |
| 2 | angvel_penalty + orientation_penalty + target_proximity + velocity_penalty | -87.67 | -53.29 | -34.38 | 1000.00 | angvel_penalty=-0.005 orientation_penalty=-0.007 target_proximity=7.147 velocity_penalty=-0.021 | no_meaningful_improvement |
| 3 | angvel_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 102.16 | 102.16 | 0.00 | 413.80 | angvel_penalty=-0.003 landing_reward=0.357 orientation_penalty=-0.001 target_proximity=-0.138 velocity_penalty=-0.029 | new_best |
| 4 | angvel_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 238.20 | 238.20 | 0.00 | 397.70 | angvel_penalty=-0.002 landing_reward=0.550 orientation_penalty=-0.001 target_proximity=-0.024 velocity_penalty=-0.015 | target_solved_new_best |
| 5 | angvel_penalty + engine_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 240.86 | 240.86 | 0.00 | 468.20 | angvel_penalty=-0.002 engine_penalty=-0.001 landing_reward=0.512 orientation_penalty=-0.001 target_proximity=-0.025 | target_solved_new_best |
| 6 | angvel_penalty + engine_penalty + landing_reward + orientation_penalty + target_proximity + velocity_penalty | 93.24 | 240.86 | -147.62 | 931.45 | angvel_penalty=-0.002 engine_penalty=-0.001 landing_reward=0.091 orientation_penalty=-0.001 target_proximity=-0.017 | stop_after_solved_drop_keep_best |
