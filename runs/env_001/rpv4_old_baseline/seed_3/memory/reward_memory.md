# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | unknown | 151.32 | 151.32 | 0.00 | 953 | contact_bonus=61.585 progress=13.733 angle_penalty=-2.758 speed_penalty=-2.049 | new_best |
| 2 | orientation + progress + soft_landing + velocity_damping | -27.07 | 151.32 | -178.39 | 1000.00 | orientation=-0.045 progress=0.001 soft_landing=2.190 velocity_damping=-0.080 | no_meaningful_improvement |
| 3 | approach_vel + landing + near_speed_penalty + orientation + progress | -33.41 | 151.32 | -184.73 | 1000.00 | approach_vel=0.054 landing=22.433 near_speed_penalty=-0.012 orientation=-0.000 progress=0.002 | no_meaningful_improvement |
| 4 | landing + orientation + progress + speed_penalty_global + speed_penalty_near + vert_speed_penalty | -158.53 | 151.32 | -309.85 | 533.00 | landing=22.374 orientation=-0.000 progress=0.008 speed_penalty_global=-0.024 speed_penalty_near=-0.281 | unsolved_high_achievement_continue_from_best |
| 5 | contact_bonus + orientation + progress + proximity + soft_landing + speed_penalty_global | -46.66 | 151.32 | -197.98 | 1000.00 | contact_bonus=2.604 orientation=-0.000 progress=0.059 proximity=1.118 soft_landing=1.609 | no_meaningful_improvement |
| 6 | contact_encouragement + engine_penalty + orientation + progress + proximity + soft_landing | -87.87 | 151.32 | -239.19 | 773.65 | contact_encouragement=1.310 engine_penalty=-0.035 orientation=-0.001 progress=0.104 proximity=0.177 | no_meaningful_improvement |
| 7 | contact_reward + engine_penalty + orientation + progress + proximity + soft_landing | -134.18 | 151.32 | -285.50 | 1000.00 | contact_reward=3.483 engine_penalty=-0.001 orientation=-0.000 progress=0.055 proximity=0.168 | unsolved_high_achievement_continue_from_best |
| 8 | contact_reward + engine_penalty + height_penalty + landing_bonus + orientation + progress | -114.53 | 151.32 | -265.85 | 68.35 | contact_reward=0.428 engine_penalty=-0.001 height_penalty=-1.853 landing_bonus=0.896 orientation=-0.001 | no_meaningful_improvement |
| 9 | contact_reward + engine_penalty + height_cost + landing_bonus + orientation_cost + progress | -131.94 | 151.32 | -283.26 | 465.40 | contact_reward=12.677 engine_penalty=-0.001 height_cost=-0.063 landing_bonus=63.935 orientation_cost=-0.001 | no_meaningful_improvement |
| 10 | contact_reward + engine_penalty + height_cost + landing_bonus + orientation_cost + position_reward | -626.75 | 151.32 | -778.07 | 88.95 | contact_reward=0.070 engine_penalty=-0.002 height_cost=-0.033 landing_bonus=0.044 orientation_cost=-0.007 | unsolved_high_achievement_continue_from_best |
