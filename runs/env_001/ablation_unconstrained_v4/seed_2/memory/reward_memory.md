# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_establishment + goal_proximity + soft_landing_stabilization + upright_attitude | -111.84 | -111.84 | 0.00 | 68.50 | contact_establishment=0.050 goal_proximity=-1.075 soft_landing_stabilization=-0.133 upright_attitude=-0.041 | new_best |
| 2 | att_penalty + contact_bonus + horizontal_penalty + progress + vel_penalty | -21.25 | -21.25 | 0.00 | 1000.00 | att_penalty=-0.008 contact_bonus=0.864 horizontal_penalty=-0.019 progress=0.003 vel_penalty=-0.009 | new_best |
| 3 | alignment + att_penalty + contact_reward + height_near_reward + horizontal_penalty + progress | 71.06 | 71.06 | 0.00 | 977.60 | alignment=0.469 att_penalty=-0.003 contact_reward=0.868 height_near_reward=0.293 horizontal_penalty=-0.004 | new_best |
| 4 | alignment + any_contact_reward + att_penalty + height_near_reward + progress + stable_contact_reward | -8.16 | 71.06 | -79.22 | 1000.00 | alignment=0.465 any_contact_reward=0.462 att_penalty=-0.004 height_near_reward=0.239 progress=0.010 | no_meaningful_improvement |
| 5 | alignment + any_contact_reward + att_penalty + height_near_reward + horizontal_penalty + progress | -69.73 | 71.06 | -140.79 | 937.95 | alignment=0.424 any_contact_reward=1.973 att_penalty=-0.006 height_near_reward=1.958 horizontal_penalty=-0.041 | no_meaningful_improvement |
| 6 | alignment + any_contact_reward + att_penalty + height_near_reward + horizontal_penalty + progress | -139.01 | 71.06 | -210.07 | 536.45 | alignment=0.077 any_contact_reward=0.936 att_penalty=-0.004 height_near_reward=0.278 horizontal_penalty=-0.014 | unsolved_high_achievement_continue_from_best |
| 7 | alignment + angvel_penalty + att_penalty + contact_reward + down_penalty + height_near | -119.60 | 71.06 | -190.66 | 685.75 | alignment=0.333 angvel_penalty=-0.126 att_penalty=-0.101 contact_reward=0.632 down_penalty=-0.039 | no_meaningful_improvement |
| 8 | alignment + angvel_penalty + att_penalty + contact_reward + down_penalty + height_near | -24.74 | 71.06 | -95.80 | 1000.00 | alignment=1.454 angvel_penalty=-0.062 att_penalty=-0.097 contact_reward=1.577 down_penalty=-0.116 | no_meaningful_improvement |
| 9 | alignment + angvel_penalty + att_penalty + contact_reward + descend_vel + down_penalty | -205.04 | 71.06 | -276.10 | 785.85 | alignment=0.726 angvel_penalty=-0.080 att_penalty=-0.089 contact_reward=3.454 descend_vel=0.019 | unsolved_high_achievement_continue_from_best |
| 10 | alignment + angvel_penalty + att_penalty + contact_reward + height_reward + lat_penalty | -296.34 | 71.06 | -367.40 | 494.25 | alignment=1.052 angvel_penalty=-0.157 att_penalty=-0.405 contact_reward=3.896 height_reward=5.915 | no_meaningful_improvement |
