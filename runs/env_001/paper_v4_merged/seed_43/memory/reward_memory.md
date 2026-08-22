# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angle_hinge + progress_shaping | -117.88 | -117.88 | 0.00 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 progress_shaping=0.015 | new_best |
| 2 | action_cost + angle_hinge + danger_penalty + progress_shaping | -117.48 | -117.48 | 0.00 | 68.35 | action_cost=-0.002 angle_hinge=-0.001 danger_penalty=-0.002 progress_shaping=0.015 | new_best |
| 3 | action_cost + angle_hinge + landing_contact_reward + progress_shaping | -122.17 | -117.48 | -4.69 | 68.30 | action_cost=-0.001 angle_hinge=-0.001 landing_contact_reward=0.003 progress_shaping=0.015 | no_meaningful_improvement |
| 4 | action_cost + landing_contact_reward + landing_speed_gate + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 landing_speed_gate=0.879 progress_shaping=0.014 shaped_progress=0.011 | new_best |
| 5 | action_cost + landing_contact_reward + progress_shaping + shaped_progress | -87.19 | -87.19 | 0.00 | 143.70 | action_cost=-0.002 landing_contact_reward=0.007 progress_shaping=0.014 shaped_progress=0.011 | no_meaningful_improvement |
| 6 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -114.35 | -87.19 | -27.16 | 68.35 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.015 shaped_progress=0.012 | no_meaningful_improvement |
| 7 | action_cost + angle_hinge_penalty + landing_contact_reward + progress_shaping + shaped_progress | -105.53 | -87.19 | -18.34 | 71.20 | action_cost=-0.001 angle_hinge_penalty=-0.000 landing_contact_reward=0.003 progress_shaping=0.008 shaped_progress=0.007 | unsolved_stagnation_fresh_restart |
| 8 | angle_penalty + fuel_cost + progress_reward + soft_landing_bonus + speed_penalty | -124.39 | -87.19 | -37.20 | 84.45 | angle_penalty=-0.002 fuel_cost=-0.005 progress_reward=0.009 soft_landing_bonus=0.004 speed_penalty=-0.015 | no_meaningful_improvement |
| 9 | action_cost + contact_factor + gate_angle + progress + shaped_progress + speed_penalty | -24.05 | -24.05 | 0.00 | 980.75 | action_cost=-0.008 contact_factor=0.696 gate_angle=0.747 progress=0.003 shaped_progress=0.001 | new_best |
| 10 | action_cost + contact_factor + gate_angle + progress + progress_bonus + shaped_progress | -2081.10 | -24.05 | -2057.05 | 696.75 | action_cost=-0.005 contact_factor=0.405 gate_angle=0.499 progress=0.005 progress_bonus=0.087 | no_meaningful_improvement |
