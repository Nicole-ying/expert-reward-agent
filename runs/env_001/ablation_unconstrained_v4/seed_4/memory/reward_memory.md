# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_vel_penalty + progress + stable_bonus + upright_penalty | -101.89 | -101.89 | 0.00 | 68.70 | landing_vel_penalty=-0.043 progress=0.016 stable_bonus=0.006 upright_penalty=-0.011 | new_best |
| 2 | hspeed_penalty + landing_vel_penalty + progress + upright_penalty + x_center_penalty | -110.66 | -101.89 | -8.77 | 68.45 | hspeed_penalty=-0.076 landing_vel_penalty=-0.044 progress=0.081 upright_penalty=-0.017 x_center_penalty=-0.071 | no_meaningful_improvement |
| 3 | contact_reward + progress | -0.24 | -0.24 | 0.00 | 483.85 | contact_reward=8.206 progress=0.032 | new_best |
| 4 | ang_penalty + contact_reward + progress + vel_penalty | 140.27 | 140.27 | 0.00 | 463.85 | ang_penalty=0.008 contact_reward=7.215 progress=0.065 vel_penalty=0.005 | new_best |
| 5 | ang_penalty + contact_reward + fuel_penalty + progress + vel_penalty | -125.82 | 140.27 | -266.09 | 68.35 | ang_penalty=0.017 contact_reward=0.159 fuel_penalty=0.053 progress=0.401 vel_penalty=0.988 | no_meaningful_improvement |
| 6 | contact_reward + progress | -78.74 | 140.27 | -219.01 | 98.55 | contact_reward=0.184 progress=0.010 | no_meaningful_improvement |
| 7 | angle_penalty + contact_reward + fuel_penalty + progress + speed_penalty | 11.67 | 140.27 | -128.60 | 602.85 | angle_penalty=-0.000 contact_reward=1.281 fuel_penalty=-0.013 progress=0.128 speed_penalty=-0.023 | unsolved_high_achievement_continue_from_best |
| 8 | angle_penalty + contact_reward + progress + speed_penalty | -59.39 | 140.27 | -199.66 | 988.90 | angle_penalty=-0.002 contact_reward=14.614 progress=0.004 speed_penalty=-0.002 | no_meaningful_improvement |
| 9 | contact_reward + distance_penalty + fuel_penalty + progress | -115.86 | 140.27 | -256.13 | 68.40 | contact_reward=0.185 distance_penalty=-0.097 fuel_penalty=-0.003 progress=0.162 | no_meaningful_improvement |
| 10 | angle_penalty + contact_reward + progress + speed_penalty | -82.84 | 140.27 | -223.11 | 1000.00 | angle_penalty=-0.138 contact_reward=4.726 progress=0.000 speed_penalty=-0.175 | unsolved_high_achievement_continue_from_best |
