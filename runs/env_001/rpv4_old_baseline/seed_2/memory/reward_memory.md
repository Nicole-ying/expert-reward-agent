# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | proximity_reward + velocity_penalty + angle_penalty + landing_bonus | -108.86 | -108.86 | 0.00 | 68 | velocity_penalty=-66.013 proximity_reward=-37.640 landing_bonus=7.070 angle_penalty=-0.032 | new_best |
| 2 | proximity_reward + speed_reward + angle_reward + landing_bonus | -64.69 | -64.69 | 0.00 | 1000 | angle_reward=798.124 speed_reward=690.654 proximity_reward=-47.048 landing_bonus=0.000 | new_best |
| 3 | proximity_reward + height_reward + contact_reward + speed_reward + angle_reward + landing_bonus | -80.09 | -64.69 | -15.40 | 1000 | proximity_reward=983.148 angle_reward=645.189 speed_reward=533.347 height_reward=502.390 | no_meaningful_improvement |
| 4 | shaping_reward + contact_reward + landing_bonus + time_penalty | -91.71 | -64.69 | -27.02 | 1000 | time_penalty=-10.000 shaping_reward=0.019 contact_reward=0.000 landing_bonus=0.000 | no_meaningful_improvement |
| 5 | descent_reward + horiz_penalty + vel_penalty + orient_penalty + angvel_penalty + contact_reward + landing_bonus + time_penalty | -72.18 | -64.69 | -7.49 | 77 | landing_bonus=60.000 contact_reward=9.700 vel_penalty=-9.097 horiz_penalty=-2.744 | no_meaningful_improvement |
| 6 | descent_shaping + horiz_penalty + vx_penalty + orient_penalty + angvel_penalty + contact_reward + landing_bonus + time_penalty | -107.51 | -64.69 | -42.83 | 798 | landing_bonus=925.000 contact_reward=659.000 engine_penalty=-135.396 shaping_reward=59.510 | no_meaningful_improvement |
| 7 | state_goodness + contact_reward + descent_bonus + time_penalty | -23.83 | -23.83 | 0.00 | 1000 | state_goodness=8039.785 time_penalty=-20.000 descent_bonus=1.273 contact_reward=0.000 | new_best |
| 8 | descent_reward + proximity_penalty + stability_reward + fuel_penalty + time_penalty | -68.15 | -23.83 | -44.32 | 713 | stability_reward=360.234 fuel_penalty=-35.403 proximity_penalty=-31.232 time_penalty=-7.128 | no_meaningful_improvement |
| 9 | approach_reward + speed_penalty + landing_reward + fuel_penalty + time_penalty | -10.11 | -10.11 | 0.00 | 694 | landing_reward=173.608 fuel_penalty=-25.070 time_penalty=-6.939 speed_penalty=-6.259 | new_best |
| 10 | approach_reward + fuel_penalty + landing_reward + speed_penalty | -119.42 | -10.11 | -109.31 | 804.00 | approach_reward=0.110 fuel_penalty=-0.001 landing_reward=17.673 speed_penalty=-0.120 | no_meaningful_improvement |
