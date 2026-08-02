# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_penalty + progress_delta + speed_penalty | 128.47 | 128.47 | 0.00 | 1000.00 | orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.002 | new_best |
| 2 | contact_reward + orientation_penalty + progress_delta + speed_penalty | 144.81 | 144.81 | 0.00 | 960.10 | contact_reward=0.129 orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.001 | new_best |
| 3 | contact_reward + landing_progress + orientation_penalty + progress_delta + speed_penalty | 141.28 | 144.81 | -3.53 | 897.70 | contact_reward=0.135 landing_progress=0.075 orientation_penalty=-0.001 progress_delta=0.002 speed_penalty=-0.001 | no_meaningful_improvement |
| 4 | angle_penalty + angvel_penalty + completion_proxy + progress_delta + speed_penalty | 195.26 | 195.26 | 0.00 | 754.65 | angle_penalty=-0.003 angvel_penalty=-0.001 completion_proxy=0.492 progress_delta=0.016 speed_penalty=-0.007 | new_best |
| 5 | angle_penalty + angvel_penalty + completion_proxy + progress_delta + speed_penalty | 186.99 | 195.26 | -8.27 | 750.35 | angle_penalty=-0.002 angvel_penalty=-0.001 completion_proxy=0.591 progress_delta=0.012 speed_penalty=-0.005 | no_meaningful_improvement |
| 6 | angle_penalty + angvel_penalty + completion_proxy + engine_penalty + progress_delta + speed_penalty | -111.41 | 195.26 | -306.67 | 68.45 | angle_penalty=-0.001 angvel_penalty=-0.004 completion_proxy=0.005 engine_penalty=-0.006 progress_delta=0.081 | no_meaningful_improvement |
| 7 | angle_penalty + angvel_penalty + boundary_warning + contact_reward + landing_bonus + progress_delta | 142.74 | 195.26 | -52.53 | 885.15 | angle_penalty=-0.005 angvel_penalty=-0.001 boundary_warning=-0.018 contact_reward=0.360 landing_bonus=0.279 | unsolved_high_achievement_continue_from_best |
| 8 | angle_penalty + angvel_penalty + completion_bonus + progress + speed_penalty | -58.91 | 195.26 | -254.18 | 1000.00 | angle_penalty=-0.008 angvel_penalty=-0.001 completion_bonus=4.515 progress=0.017 speed_penalty=-0.014 | no_meaningful_improvement |
| 9 | angle_penalty + angvel_penalty + boundary_penalty + completion + contact_reward + progress | -45.07 | 195.26 | -240.33 | 1000.00 | angle_penalty=-0.027 angvel_penalty=-0.004 boundary_penalty=-0.001 completion=2.269 contact_reward=0.190 | no_meaningful_improvement |
| 10 | angle_penalty + angvel_penalty + completion + contact_reward + progress + speed_penalty | -118.50 | 195.26 | -313.76 | 1000.00 | angle_penalty=-0.008 angvel_penalty=-0.001 completion=0.625 contact_reward=0.004 progress=0.006 | unsolved_high_achievement_continue_from_best |
